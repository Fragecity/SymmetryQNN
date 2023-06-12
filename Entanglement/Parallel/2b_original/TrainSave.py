import pickle
import pennylane as qml
import pennylane.numpy as np
from Utils import reflection
from Ansatzs.EasyPQC import EasyPQC
from Train_steps import trainAndCompare

#! ------------------------- Configs ------------------------- #

ds_loadPath = './data/test_ds.pkl'
rs_savePath = './data/test_2b_rs.pkl'

MAX_ITER = 30
ETA = 0.03
LAMD = 1

#! ------------------------- Data & Symmetry ------------------------- #

with open(ds_loadPath, 'rb') as f:
    [NUM_QUBIT, train_set, test_set, symmetry] = pickle.load(f)
NUM_QUBIT = 2* NUM_QUBIT

#! ------------------------- QNN ------------------------- #

dev = qml.device('default.mixed', wires=NUM_QUBIT)
O = symmetry.observable

def enc_func(x, paras):
    qml.QubitDensityMatrix(x, wires=[0,1])

def ansatz(para):
    qml.U3(para[0], para[1], para[2], 0)
    qml.U3(para[3], para[4], para[5], 1)
    qml.CNOT([0,1])
    qml.U3(para[6], para[7], para[8], 0)
    qml.U3(para[9], para[10], para[11], 1)
    qml.QubitUnitary(reflection(para[12]), 0)
    qml.QubitUnitary(reflection(para[13]), 1)

def ansatzOb():
    return qml.Hermitian(O, wires=[0,1])

pqc = EasyPQC(dev, enc_func, ansatz, ansatzOb, ansatzOb)

#! ------------------------- Training & Save ------------------------- #

para_init = np.random.random(14, requires_grad=True) *np.pi/2
cst_lst, cstG_lst, para_lst, paraG_lst = trainAndCompare(pqc, para_init, symmetry, train_set, test_set, ETA, LAMD, MAX_ITER)

with open(rs_savePath, 'wb') as f:
    pickle.dump([cst_lst, cstG_lst, para_lst, paraG_lst], f)



