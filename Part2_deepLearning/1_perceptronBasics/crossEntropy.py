import numpy as np

def cross_entropy(Y, P):
    """
    that takes as input two lists Y, P, and returns the float corresponding to their cross-entropy.
    :param Y:
    :param P:
    :return:
    """
    Y = np.float_(Y)
    P = np.float_(P)
    ln_p = np.log(P)
    ln_np = np.log(1-P)
    cross_entropy = -1 * sum(Y*ln_p + (1-Y)*ln_np)
    return cross_entropy


Y = [1,1,0]
P=[0.8,0.9,0.2]
v = cross_entropy(Y, P)
print('cross entropy: ' , v)