import pennylane as qml
import pennylane.numpy as np
import pickle
import os
from time import time
from Utils import dataRepo_path
from Train_steps import trainAndCompare
from Ansatzs.ZYCX import ZYCXAnsatz
from Ansatzs.Y import YAnsatz
from Symmetries.AmpSwapSymmetry import AmpSwapSymmetry
from Symmetries.SwapSymmetry import SwapSymmetry

num_bits = 1
dev = qml.device('default.mixed', wires=num_bits)


# TODO: model selection part could be more concise
# #* model selection: ZYCX
# NUM_LAYER = 28
# pqc = ZYCXAnsatz(dev, num_bits, NUM_LAYER)
# #* parameter initialization
# NUM_PARA = (pqc.num_layers + 2) * pqc.num_bits
# para_init = np.random.random(NUM_PARA, requires_grad=True) * 2 * np.pi

#* model selection: Y
NUM_LAYER = 1
pqc = YAnsatz(dev, num_bits, NUM_LAYER)
#* parameter initialization
NUM_PARA = 1
para_init = np.random.random(NUM_PARA, requires_grad=True) * 2 * np.pi


#* symmetry group
# TODO: symmetry group could be related to the model
symmetry = AmpSwapSymmetry(num_bits)


#* data loading
data_path = os.path.join(dataRepo_path, 'xpy_nt30_nv90_3.pkl')
with open(data_path, 'rb') as f:
        [train_data, test_data] = pickle.load(f)
        # [num_bits, train_data, test_data, symmetry] = pickle.load(f)


#* guidance weight
LAMD = 0.5


#* run with time recording
ETA = 0.03 # step_size here
MAX_ITER = 30
record = []
t0 = time()
trainAndCompare(record, pqc, para_init, symmetry, train_data, test_data, ETA, LAMD, MAX_ITER)
t1 = time()
print(f"Run time: {(t1-t0)/60 : .2f} min")