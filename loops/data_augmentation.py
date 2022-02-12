import numpy as np
import random

def augment_switch(data):
    new_data = []
    for statement in data:
        new_data.append(statement)
        for i in range(3, len([j for j in statement if j != 0]), 2):
            statement_copy = statement.copy()
            temp = statement_copy[1:3]
            statement_copy[1:3] = statement_copy[i:i+2]
            statement_copy[i:i+2] = temp
            new_data.append(statement_copy)
    # new_data = np.array(new_data)
    return new_data


def sample_data(data, sample_rate):
    new_data = random.sample(data, int(len(data)*sample_rate))
    return new_data


def augment_decompose(data):
    '''
    only keep the qualifiers
    '''
    new_data = []
    for statement in data:
        new_data.append(statement)
        for i in range(3, len([j for j in statement if j != 0]), 2):
            statement_copy = statement.copy()
            statement_copy[1:3] = statement_copy[i:i+2]
            statement_copy[i:i+2] = [0]*2
            new_data.append(statement_copy)
    # new_data = np.array(new_data)
    return new_data
