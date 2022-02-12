import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict
from .gnn_encoder import StarEEncoder, StarEBase
from utils.utils_gcn import get_param, ccorr, rotate, softmax


class StarE_Transformer(StarEEncoder):
    model_name = 'StarE_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.p_option = config['STAREARGS']['POS_OPTION']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.LayerNorm = nn.LayerNorm(config['EMBEDDING_DIM'])

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals, obj=None):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True, obj)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        if self.config['STAREARGS']['SEP_ENT_EMBEDDING']:
            x = torch.mm(x, all_ent[:, :all_ent.shape[1]//2].transpose(1, 0))
        else:
            x = torch.mm(x, all_ent.transpose(1, 0))
            # # jump embed
            # x = torch.mm(x, self.init_embed.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class StarE_ObjectMask_Transformer(StarEEncoder):
    model_name = 'StarE_ObjectMask_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):

        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']

        self.object_mask_emb = torch.nn.Parameter(torch.randn(1, self.emb_dim,dtype=torch.float32),True)
        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'], self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, e1_embed, rel_embed, obj_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        obj_embed = obj_embed.view(-1,1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, obj_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp        # 14, 128, 200

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)


        # bs*emb_dim , ......, bs*6*emb_dim
        object_mask = self.object_mask_emb.repeat(sub.shape[0], 1)
        ins = torch.zeros((sub.shape), dtype=torch.bool, device=self.device)
        mask = torch.cat((mask[:, :2], ins.unsqueeze(1), mask[:, 2:]), axis=1)

        stk_inp = self.concat(sub_emb, rel_emb, object_mask, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)[2] # to get the object position

        x = self.fc(x)

        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class StarE_Transformer_TripleBaseline(StarEEncoder):
    model_name = 'StarE_Transformer_Triple_Baseline'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb)
        mask = mask[:, :2]

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class Transformer_Statements_mask(StarEBase):
    """Baseline for Transformer decoder only model w/o starE encoder
    with well chosen setting
    """

    def __init__(self, config: dict):
        super().__init__(config)

        #self.emb_dim = config['EMBEDDING_DIM']
        self.entities = get_param((self.num_ent+1, self.emb_dim), norm=False) # final ind for [MASK]
        self.relations = get_param((2 * self.num_rel, self.emb_dim), norm=False)

        self.model_name = 'Transformer_Statements_mask'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']

        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat
        self.device = config['DEVICE']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden,
                                                 config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'], self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

        # added
        self.act = torch.tanh if 'ACT' not in config['STAREARGS'].keys() \
            else config['STAREARGS']['ACT']
        self.bn = torch.nn.BatchNorm1d(config['EMBEDDING_DIM'])
        self.LayerNorm = nn.LayerNorm(config['EMBEDDING_DIM'])
        self.LayerNorm1 = nn.LayerNorm(config['EMBEDDING_DIM'])
        self.LayerNorm_combine = nn.LayerNorm(config['EMBEDDING_DIM'])

    def concat(self, e1_embed, rel_embed, obj_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        obj_embed = obj_embed.view(-1,1, self.emb_dim)

        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, obj_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals, obj=None):

        self.score_list = [0., 0.]

        entitiy_embeddings = self.entities
        entitiy_embeddings = self.LayerNorm(entitiy_embeddings)
        entitiy_embeddings = self.feature_drop(entitiy_embeddings)
        relation_embeddings = self.relations
        relation_embeddings = self.LayerNorm1(relation_embeddings)

        sub_emb = torch.index_select(entitiy_embeddings, 0, sub)
        rel_emb = torch.index_select(relation_embeddings, 0, rel)

        if obj is None:
            obj = torch.tensor([self.num_ent]).long().to(sub.device).repeat(sub.shape)
            mask_ind = torch.tensor([2]).long().to(sub.device).repeat(sub.shape)
        else:
            statements = torch.cat([sub.reshape(-1,1),rel.reshape(-1,1),obj.reshape(-1,1),quals], dim=1)
            mask_ind = (statements==self.num_ent).nonzero()[:,1]
            # if multiple entities are masked:
            # ind_aux = torch.arange(statements.shape[1], 0, -1).to(statements.device)
            # ind_aux = (statements==self.num_ent) * ind_aux
            # mask_ind = torch.argmax(ind_aux, dim=1)
        obj_emd = torch.index_select(entitiy_embeddings, 0, obj)

        # print(statements==self.num_ent)
        quals_ents = quals[:, 1::2].reshape(1, -1).squeeze(0)
        quals_rels = quals[:, 0::2].reshape(1, -1).squeeze(0)
        qual_obj_emb = torch.index_select(entitiy_embeddings, 0, quals_ents)
        qual_rel_emb = torch.index_select(relation_embeddings, 0, quals_rels)
        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        # so we first initialize with False
        mask = torch.zeros((sub.shape[0], quals.shape[1] + 3)).bool().to(self.device)
        # and put True where qual entities and relations are actually padding index 0
        mask[:, 3:] = quals == 0

        stk_inp = self.concat(sub_emb, rel_emb, obj_emd, qual_rel_emb, qual_obj_emb)

        if self.positional:
            # positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            qual_ind = 5
            positions_main = torch.arange(qual_ind, dtype=torch.long, device=self.device)
            positions_qual = torch.arange(qual_ind-2, qual_ind, dtype=torch.long, device=self.device).repeat((stk_inp.shape[0]-qual_ind)//2)
            positions = torch.cat([positions_main, positions_qual]).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        # stk_inp = self.LayerNorm_combine(stk_inp)
        # stk_inp = self.hidden_drop2(stk_inp)

        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        x = x[mask_ind, torch.arange(x.shape[1]).to(x.device)]

        x = self.fc(x)

        x = torch.mm(x, entitiy_embeddings[:-1].transpose(1, 0)) # final ind for mask embedding
        # x = torch.mm(x, self.entities[:-1].transpose(1, 0)) # final ind for mask embedding
        # x = torch.mm(x, self.feature_drop(self.LayerNorm(entitiy_embeddings))[:-1].transpose(1, 0)) # final ind for mask embedding

        score = torch.sigmoid(x)
        return score

