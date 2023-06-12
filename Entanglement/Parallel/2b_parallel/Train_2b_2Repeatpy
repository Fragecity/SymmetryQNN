import pennylane.numpy as np
import pennylane as qml
import pickle, copy
from Ansatzs.EasyPQC import EasyPQC
from Train_steps import trainAndCompare
from Utils import reflection

#! ------------------------- Configs ------------------------- #

ds_loadPath = './data/test_ds.pkl'
rs_savePath = './data/test_2b_rs.pkl'

MAX_ITER = 30
ETA = 0.015
LAMD = 1
PRETRAIN_ITER = 4

#! ------------------------- Data & Symmetry ------------------------- #

with open(ds_loadPath, 'rb') as f:
    [NUM_QUBIT, train_set, test_set, symmetry] = pickle.load(f)
NUM_QUBIT = 2* NUM_QUBIT

#! ------------------------- QNN ------------------------- #

dev = qml.device('default.mixed', wires=4*NUM_QUBIT)
O = symmetry.observable

def enc_func(x, paras):
    qml.QubitDensityMatrix(x, wires=[0,1])
    qml.QubitDensityMatrix(x, wires=[2,3])
    qml.QubitDensityMatrix(x, wires=[4,5])
    qml.QubitDensityMatrix(x, wires=[6,7])

def layer(para):
    qml.CNOT([0,1])
    qml.CNOT([2,3])
    qml.CNOT([4,5])
    qml.CNOT([6,7])
    qml.QubitUnitary(reflection(para[0]), 0)
    qml.QubitUnitary(reflection(para[1]), 1)
    qml.QubitUnitary(reflection(para[2]), 2)
    qml.QubitUnitary(reflection(para[3]), 3)
    qml.QubitUnitary(reflection(para[4]), 4)
    qml.QubitUnitary(reflection(para[5]), 5)
    qml.QubitUnitary(reflection(para[6]), 6)
    qml.QubitUnitary(reflection(para[7]), 7)

def ansatz(para):
    LAYERS = 3
    for ind_layer in range(LAYERS):
        layer(para[ind_layer: ind_layer + 8])

def ansatzOb():
    return qml.Hermitian(O, wires=range(8))

pqc = EasyPQC(dev, enc_func, ansatz, ansatzOb, ansatzOb)

#! ------------------------- Training & Save ------------------------- #

para_init = np.array([1.97294487, 1.6568124 , 2.50297357, 1.04580963, 0.40185501,
        1.19904697, 1.82736537, 2.56028965, 1.5795827 , 2.39021789,
        2.6468656 , 1.08283574, 1.48595238, 0.15473284, 0.15853485,
        1.52222387, 1.98900387, 1.89221966, 2.24512592, 0.10475985,
        1.42263824, 0.54568072, 3.10190585, 0.86365369], requires_grad=True)

cst_lst, cstG_lst, para_lst, paraG_lst = trainAndCompare(pqc, para_init, symmetry, train_set, test_set, 
                                                         ETA, LAMD, MAX_ITER, preTrain=True, PRETRAIN_ITER = PRETRAIN_ITER)
        
with open(rs_savePath, 'wb') as f:
    pickle.dump([cst_lst, cstG_lst, para_lst, paraG_lst], f)
