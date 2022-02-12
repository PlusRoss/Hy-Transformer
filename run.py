import os
os.environ['MKL_NUM_THREADS'] = '1'

from functools import partial
import random
import wandb
import sys
import collections
from warmup_scheduler import GradualWarmupScheduler


# Local imports
from data_loaders.data_manager import DataManager
from utils.utils import *
from utils.utils_mytorch import FancyDict, parse_args, BadParameters, mt_save_dir
from loops.evaluation import EvaluationBenchGNNMultiClass, evaluate_pointwise
from loops.evaluation import acc, mrr, mr, hits_at
from models.models import StarE_ConvKB_Statement, StarE_Transformer_Triples, Transformer_Baseline
from models.models_statements import StarE_Transformer, StarE_ObjectMask_Transformer, \
    StarE_Transformer_TripleBaseline, Transformer_Statements_mask
from loops.corruption import Corruption
from loops.sampler import MultiClassSampler, MultiClassSampler_mask
from loops.loops import training_loop_gcn
from loops.data_augmentation import augment_switch, augment_decompose, sample_data

"""
    CONFIG Things
"""

DEFAULT_CONFIG = {
    'DATA_AUG': 'None',
    'DATA_SAMPLE': 1.0,
    'SEED': 1996,
    'QUAL_PROB': 1.0,

    'BATCH_SIZE': 128,
    'DATASET': 'wd50k',
    'DEVICE': 'cpu',
    'EMBEDDING_DIM': 200,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 401,
    'EVAL_EVERY': 5,
    'LEARNING_RATE': 0.0001,
    'WEIGHT_DECAY': 0,

    'MAX_QPAIRS': 15,
    'MODEL_NAME': 'hy-transformer',
    'CORRUPTION_POSITIONS': [0, 2],

    # # not used for now
    # 'MARGIN_LOSS': 5,
    # 'NARY_EVAL': False,
    # 'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    # 'NEGATIVE_SAMPLING_TIMES': 10,
    # 'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    # 'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    # 'NUM_FILTER': 5,
    # 'PROJECT_QUALIFIERS': False,
    # 'PRETRAINED_DIRNUM': '',
    # 'RUN_TESTBENCH_ON_TRAIN': False,
    # 'SAVE': False,
    # 'SELF_ATTENTION': 0,
    # 'SCORING_FUNCTION_NORM': 1,

    # important args
    'SAVE': False,
    'STATEMENT_LEN': -1,
    'USE_TEST': True,
    'WANDB': False,
    'LABEL_SMOOTHING': 0.1,
    'SAMPLER_W_QUALIFIERS': True,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': True,  # should be false for WikiPeople and JF17K for their original data
    'SUBTYPE': 'statements',

    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': True
}

STAREARGS = {
    # new added:
    'WEIGHT_TRANS': True,
    'SEP_ENT_EMBEDDING': False,
    'MASK_EDGE': True,

    'ENCODER': "StarE",
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 200,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'BIAS': False,
    'OPN': 'rotate',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
    'QUAL_OPN': 'rotate',
    'QUAL_N': 'sum',  # or mean
    'SUBBATCH': 0,
    'QUAL_REPR': 'sparse',  # sparse or full. Warning: full is 10x slower
    'ATTENTION': False,
    'ATTENTION_HEADS': 4,
    'ATTENTION_SLOPE': 0.2,
    'ATTENTION_DROP': 0.1,
    'HID_DROP2': 0.1,

    # For ConvE Only
    'FEAT_DROP': 0.3,
    'N_FILTERS': 200,
    'KERNEL_SZ': 7,
    'K_W': 10,
    'K_H': 20,

    # For Transformer
    'T_LAYERS': 2,
    'T_N_HEADS': 4,
    'T_HIDDEN': 512,
    'POSITIONAL': True,
    'POS_OPTION': 'default',
    'TIME': False,
    'POOLING': 'avg'

}

DEFAULT_CONFIG['STAREARGS'] = STAREARGS

if __name__ == "__main__":

    # Get parsed arguments
    config = DEFAULT_CONFIG.copy()
    gcnconfig = STAREARGS.copy()
    parsed_args = parse_args(sys.argv[1:])
    print(parsed_args)

    # Superimpose this on default config
    for k, v in parsed_args.items():
        # If its a generic arg
        if k in config.keys():
            default_val = config[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                config[k.upper()] = needed_type(v)
            else:
                config[k.upper()] = v
        elif k.lower().startswith('gcn_') and k[4:] in gcnconfig:
            default_val = gcnconfig[k[4:].upper()]
            if default_val is not None:
                needed_type = type(default_val)
                gcnconfig[k[4:].upper()] = needed_type(v)
            else:
                gcnconfig[k[4:].upper()] = v

        else:
            config[k.upper()] = v

    # Clamp the randomness
    seed = config['SEED'] #42 132
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    config['STAREARGS'] = gcnconfig

    """
        Custom Sanity Checks
    """
    # If we're corrupting something apart from S and O
    if max(config['CORRUPTION_POSITIONS']) > 2:
        assert config['ENT_POS_FILTERED'] is False, \
            f"Since we're corrupting objects at pos. {config['CORRUPTION_POSITIONS']}, " \
            f"You must allow including entities which appear exclusively in qualifiers, too!"

    """
        Loading and preparing data

        Typically, sending the config dict, and executing the returned function gives us data,
        in the form of
            -> train_data (list of list of 43 / 5 or 3 elements)
            -> valid_data
            -> test_data
            -> n_entities (an integer)
            -> n_relations (an integer)
            -> ent2id (dictionary to interpret the data above, if needed)
            -> rel2id


    """
    data = DataManager.load(config=config)()

    # Break down the data
    try:
        train_data, valid_data, test_data, n_entities, n_relations, _, _ = data.values()
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {config['DATASET']}")

    if "mask" not in config["MODEL_NAME"]:
        if config['DATA_AUG'] == "switch":
            train_data = augment_switch(train_data)
            if config['USE_TEST']:
                valid_data = augment_switch(valid_data)
        elif config['DATA_AUG'] == "decompose":
            train_data = augment_decompose(train_data)
            if config['USE_TEST']:
                valid_data = augment_decompose(valid_data)
        elif config['DATA_AUG'] == "None":
            pass
        else:
            exit("wrong config DATA_AUG")

    if config['DATA_SAMPLE'] < 1.0:
        train_data = sample_data(train_data, config['DATA_SAMPLE'])
        if config['USE_TEST']:
            valid_data = sample_data(valid_data, config['DATA_SAMPLE'])

    config['NUM_ENTITIES'] = n_entities
    config['NUM_RELATIONS'] = n_relations

    # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
    # always off for wikipeople and jf17k
    if config['DATASET'] == 'jf17k' or config['DATASET'] == 'wikipeople':
        config['ENT_POS_FILTERED'] = False

    if config['ENT_POS_FILTERED']:
        ent_excluded_from_corr = DataManager.gather_missing_entities(
            data=train_data + valid_data + test_data,
            positions=config['CORRUPTION_POSITIONS'],
            n_ents=n_entities)
    else:
        ent_excluded_from_corr = [0]

    """
     However, when we want to run a GCN based model, we also work with
            COO representations of triples and qualifiers.

            In this case, for each split: [train, valid, test], we return
            -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
            -> edge_type (n) array with [relation] corresponding to sub, obj above
            -> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edge that has quals

        So here, train_data_gcn will be a dict containing these ndarrays.
    """

   
    # Replace the data with their graph repr formats
    if config['STAREARGS']['QUAL_REPR'] == 'full':
        if config['USE_TEST']:
            train_data_gcn = DataManager.get_graph_repr(train_data + valid_data, config)
        else:
            train_data_gcn = DataManager.get_graph_repr(train_data, config)
    elif config['STAREARGS']['QUAL_REPR'] == 'sparse':
        if config['USE_TEST']:
            train_data_gcn = DataManager.get_custom_graph_repr(train_data + valid_data, config)
        else:
            train_data_gcn = DataManager.get_custom_graph_repr(train_data, config)
    else:
        print("Supported QUAL_REPR are `full` or `sparse`")
        raise NotImplementedError

    # add reciprocals to the train data
    reci = DataManager.add_reciprocals(train_data, config)
    train_data.extend(reci)
    reci_valid = DataManager.add_reciprocals(valid_data, config)
    valid_data.extend(reci_valid)
    reci_test = DataManager.add_reciprocals(test_data, config)
    test_data.extend(reci_test)

    print(f"Training on {n_entities} entities")

    """
        Make the model.
    """
    config['DEVICE'] = torch.device(config['DEVICE'])

    if config['MODEL_NAME'].lower().startswith('hy-transformer'):
        if config['SAMPLER_W_QUALIFIERS']:
            model = Transformer_Statements_mask(config)
        else:
            raise NotImplementedError
    else:
        raise BadParameters(f"Unknown Model Name {config['MODEL_NAME']}")


    model.to(config['DEVICE'])
    print("Model params: ",sum([param.nelement() for param in model.parameters()]))

    if config['OPTIMIZER'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.98),
        #                              eps=1e-09, weight_decay=config['WEIGHT_DECAY'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'],
                                     weight_decay=config['WEIGHT_DECAY'])
    else:
        print("Unexpected optimizer, we support `sgd` or `adam` at the moment")
        raise NotImplementedError


    if config['WANDB']:
        wandb.init(project="wikidata-embeddings")
        wandb.run.name = "data-{}-model-{}-aug-{}-relnorm-pos{}-seed{}-quap{}-".format(config["DATASET"],
                         config["MODEL_NAME"], config["DATA_AUG"], config["STAREARGS"]["POSITIONAL"],
                         config['SEED'], config['QUAL_PROB'])
        wandb.run.name += wandb.run.id
        for k, v in config.items():
            wandb.config[k] = v

    """
        Prepare test benches.

            When computing train accuracy (`ev_tr_data`), we wish to use all the other data
                to avoid generating true triples during corruption.
            Similarly, when computing test accuracy, we index train and valid splits
                to avoid generating negative triples.
    """
    if config['USE_TEST']:
        ev_vl_data = {'index': combine(train_data, valid_data), 'eval': combine(test_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data, valid_data), 'valid': ev_vl_data['eval']}
    else:
        ev_vl_data = {'index': combine(train_data, test_data), 'eval': combine(valid_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data), 'valid': ev_vl_data['eval']}


    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3),
                    partial(hits_at, k=5), partial(hits_at, k=10)]


    evaluation_valid = None
    evaluation_train = None

    # Saving stuff
    if config['SAVE']:
        savedir = Path(f"./models/{config['DATASET']}/{config['MODEL_NAME']}")
        if not savedir.exists(): savedir.mkdir(parents=True)
        savedir = mt_save_dir(savedir, _newdir=True)
        save_content = {'model': model, 'config': config}
    else:
        savedir, save_content = None, None

    # The args to use if we're training w default stuff
    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "train_fn": model,
        "neg_generator": Corruption(n=n_entities, excluding=[0],
                                    position=list(range(0, config['MAX_QPAIRS'], 2))),
        "device": config['DEVICE'],
        "data_fn": None,
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run if evaluation_valid else None,
        "trn_testbench": evaluation_train.run if evaluation_train else None,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": False,
        "savedir": savedir,
        "save_content": save_content,
        "qualifier_aware": config['SAMPLER_W_QUALIFIERS'],
        "grad_clipping": config['GRAD_CLIPPING'],
        "scheduler": None
    }

    if config['MODEL_NAME'].lower().startswith('hy-transformer'):
        training_loop = training_loop_gcn
        if "mask" not in config['MODEL_NAME']:
            sampler = MultiClassSampler(data= args['data']['train'],
                                        n_entities=config['NUM_ENTITIES'],
                                        lbl_smooth=config['LABEL_SMOOTHING'],
                                        bs=config['BATCH_SIZE'],
                                        with_q=config['SAMPLER_W_QUALIFIERS'])
        else:
            sampler = MultiClassSampler_mask(data= args['data']['train'],
                                        n_entities=config['NUM_ENTITIES'],
                                        lbl_smooth=config['LABEL_SMOOTHING'],
                                        bs=config['BATCH_SIZE'],
                                        with_q=config['SAMPLER_W_QUALIFIERS'],
                                        aug=(config['DATA_AUG'] != "None"),
                                        qua_prob=config['QUAL_PROB'])

        evaluation_valid = EvaluationBenchGNNMultiClass(ev_vl_data, model, bs=config['BATCH_SIZE'], metrics=eval_metrics,
                                           filtered=True, n_ents=n_entities,
                                           excluding_entities=ent_excluded_from_corr,
                                           positions=config.get('CORRUPTION_POSITIONS', None), config=config)
        args['data_fn'] = sampler.reset
        args['val_testbench'] = evaluation_valid.run
        args['trn_testbench'] = None
        if config['LR_SCHEDULER']:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
            # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)
            args['scheduler'] = scheduler

    # dummy_input = (torch.ones(512).long().cuda(), torch.ones(512).long().cuda(), torch.ones(512, 12).long().cuda())
    # torch.onnx.export(model, dummy_input, './model.onnx', input_names=["sub", "rel", "qual"])

    traces = training_loop(**args)

    with open('traces.pkl', 'wb+') as f:
        pickle.dump(traces, f)
