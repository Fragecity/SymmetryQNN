import numpy as np
import pennylane.numpy as qnp

#! --------------------------- Samples Generation --------------------------- !#
# generate samples of a,b that satisfying 0=<a,b<=1 and 0<=a+b<=1
def gen_abs_uniform(num_samples):
    # generate a linear space of numbers between 0 and 1
    lin_space = np.linspace(0, 1, num_samples)

    # initialize list to store valid pairs
    valid_pairs = []

    # check each pair of numbers
    for i in range(num_samples):
        for j in range(num_samples):
            # if the sum is less or equal than 1, add the pair to the list
            if lin_space[i] + lin_space[j] <= 1:
                valid_pairs.append((lin_space[i], lin_space[j]))

    return valid_pairs

def sigmoid(x):
    return 1 / (1 + qnp.exp(-x))
