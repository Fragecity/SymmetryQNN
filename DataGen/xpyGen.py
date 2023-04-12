import os.path
import pickle
import pennylane.numpy as np
from Utils import dataRepo_path

from Symmetries.AmpSwapSymmetry import AmpSwapSymmetry
from Symmetries.SwapSymmetry import SwapSymmetry


def a_xpydata_with_label() -> tuple:
    """case1: 2-dim data aside the line y = -x, denoted as xpy data for x + y = 0"""
    # generate x1, x2 in [-1, 1] uniformly, then to range [-0.9*pi, 0.9*pi]
    two_dim_data = (np.random.rand(2) * 2 - 1) * np.pi * 0.9
    two_dim_data.requires_grad = False
    # label = 1 if x1 + x2 > 0 else 0, which is the line y = -x
    label = two_dim_data[0] + two_dim_data[1] > 0
    return two_dim_data, label


def generate_xpydata(data_num: int) -> list:
    return [a_xpydata_with_label() for _ in range(data_num)]


def encoded_data_set(data_num):
    '''Used when parallel encoding and num_parallel = 3'''
    datas = []
    for x, label in generate_xpydata(data_num):
        data = np.append(x,x)
        data = np.append(data, x)
        datas.append((data, label))
    return datas

def ampSwapSymData():
    #* symmetry group
    num_bits = 6
    symmetry = SwapSymmetry(num_bits)  # SwapSymmetry  AmpSwapSymmetry

    #* dataset
    num_data = 30
    train_data = encoded_data_set(num_data)
    test_data = encoded_data_set(num_data*6)
    # train_data = generate_xpydata(num_data)
    # test_data = generate_xpydata(round(num_data/2))


    #* save
    save_path = os.path.join(dataRepo_path, f'swapSym_nb{num_bits}_nd{num_data}_1.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump([num_bits, train_data, test_data, symmetry], f)

# %%
if __name__ == "__main__":
    ampSwapSymData()
    print("Gen Done!")
