import pennylane.numpy as np
import pennylane as qml
import pickle
from Utils import reflection
from Ansatzs.EasyPQC import EasyPQC
from Train_steps import trainAndCompare

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

REPEAT = 3
LAYERS = 3
dev = qml.device('default.mixed', wires=REPEAT*NUM_QUBIT)
O = symmetry.observable

def enc_func(x, paras):
    for idx in range(REPEAT):
        qml.QubitDensityMatrix(x, wires=range(idx*NUM_QUBIT, idx*NUM_QUBIT+1))

def layer(para):
    qml.CNOT([0,1])
    qml.CNOT([2,3])
    qml.CNOT([4,5])
    qml.QubitUnitary(reflection(para[0]), 0)
    qml.QubitUnitary(reflection(para[1]), 1)
    qml.QubitUnitary(reflection(para[2]), 2)
    qml.QubitUnitary(reflection(para[3]), 3)
    qml.QubitUnitary(reflection(para[4]), 4)
    qml.QubitUnitary(reflection(para[5]), 5)
    
def ansatz(para):
    for ind_layer in range(LAYERS):
        layer(para[ind_layer: ind_layer + NUM_QUBIT*REPEAT])

def ansatzOb():
    return qml.Hermitian(O, wires=range(NUM_QUBIT*REPEAT))

pqc = EasyPQC(dev, enc_func, ansatz, ansatzOb, ansatzOb)

#! ------------------------- Training & Save ------------------------- #

para_init = np.random.random(REPEAT*NUM_QUBIT*LAYERS, requires_grad=True) *np.pi

cst_lst, cstG_lst, para_lst, paraG_lst = trainAndCompare(pqc, para_init, symmetry, train_set, test_set, 
                                                         ETA, LAMD, MAX_ITER, preTrain=True, PRETRAIN_ITER=PRETRAIN_ITER)

with open('res 2b3r (0).pkl', 'wb') as f:
    pickle.dump([cst_lst, cstG_lst, para_lst, paraG_lst], f)
