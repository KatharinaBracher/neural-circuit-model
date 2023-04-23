import numpy as np


def sigmoid(x):
    '''Activation function'''
    return 1 / (1 + np.exp(-x))
