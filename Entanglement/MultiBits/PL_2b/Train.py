import json, pickle
import pennylane as qml
import pennylane.numpy as np
from Ansatzs.EasyPQC import EasyPQC
from Train_steps import trainAndCompare
from Entanglement.WernerUtils import witness_PL

#! ------------------------- Configs ------------------------- #

with open('Config.json') as json_file:
    CONFIG = json.load(json_file)

ds_loadPath = './data/test_ds.pkl'
rs_savePath = './data/test_2b_rs.pkl'

#! ------------------------- Data & Symmetry ------------------------- #

with open(ds_loadPath, 'rb') as f:
    [sample_set, test_set, symmetry] = pickle.load(f)

#! ------------------------- QNN ------------------------- #

NUM_PART, NUM_LAYER = CONFIG['num_part'], CONFIG['num_layer']
dev = qml.device("default.mixed", wires=2*NUM_PART)

def enc_func(rho, parameters):
    qml.QubitDensityMatrix(rho, wires=range(2*NUM_PART))

def layer(num_bits, parameters, pattern):
    qml.broadcast(qml.RY, wires=range(num_bits), 
                  pattern="single", parameters=parameters )
    qml.broadcast(qml.CNOT, wires=range(num_bits), pattern=pattern)

def ansatz(parameters, num_layers):
    n = 2*NUM_PART
    para_arr = [parameters[i:i+n] for i in range(0, len(parameters), n)]
    para_arr.reverse()

    qml.broadcast(qml.RZ, wires=range(n), 
                  pattern="single", parameters=para_arr.pop() )
    for i in range(num_layers):
        pattern = "double" if i%2 == 0 else "double_odd"
        layer(n, para_arr.pop(), pattern)
    qml.broadcast(qml.RX, wires=range(n), 
                  pattern="single", parameters=para_arr.pop() )

def wCirOb():
    W = witness_PL(NUM_PART)
    return qml.Hermitian(np.diag(np.linalg.eigvals(W)), wires=range(2*NUM_PART))

def ansatzOb():
    return qml.expval(qml.Identity(0))

pqc = EasyPQC(dev, enc_func, ansatz, ansatzOb, wCirOb)

#! ------------------------- Training & Save ------------------------- #

num_para = (NUM_LAYER + 2) * 2*NUM_PART
para_init = np.random.random(num_para, requires_grad=True) *2*np.pi

# opt = qml.GradientDescentOptimizer(stepsize = CONFIG['eta']) #TODO: add this into train_steps inputs

cst_lst, cstG_lst, para_lst, paraG_lst = trainAndCompare(pqc, para_init, symmetry, sample_set, test_set,
                                                            CONFIG['eta'], CONFIG['lamd'], CONFIG['max_iter'])