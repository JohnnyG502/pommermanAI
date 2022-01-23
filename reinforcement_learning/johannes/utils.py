
import numpy as np


def binaryEncoding(observation, elements):
    """
    transforms into binary encoding 
    """

    binary_enc = np.zeros(shape=(11, 11, len(elements)))
    for i,v in enumerate(observation):
        for j,c in enumerate(v):
            if c in elements: binary_enc[i, j, elements.index(c)] = 1
    return binary_enc

def concatObservations(last_obs, current_obs, message_obs):
    return np.concatenate((last_obs, current_obs, message_obs), axis=2)
