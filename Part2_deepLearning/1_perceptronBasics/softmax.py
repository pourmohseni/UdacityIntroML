import numpy as np

def softmax(L):
    """
    takes as input a list of numbers, and returns the list of values given by the softmax function.
    :param L:
    :return:
    """
    exp_values = np.exp(L)
    sum_values = sum(exp_values)
    softmax_values = exp_values / sum_values
    return softmax_values

list = [2,4,6]
sm = softmax(list)
print('soft max = ' , sm)