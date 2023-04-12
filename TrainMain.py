import pennylane as qml
import pickle
import os
from time import time
from Utils import dataRepo_path
from Train_steps import trainAndCompare
from Ansatzs.ZYCX import ZYCXAnsatz


#* setting and data loading
data_path = os.path.join(dataRepo_path, 'swapSym_nb6_nd30_1.pkl')
with open(data_path, 'rb') as f:
        [num_bits, train_data, test_data, symmetry] = pickle.load(f)

dev = qml.device('default.mixed', wires=num_bits)

#* circuit selection
NUM_LAYER = 28
pqc = ZYCXAnsatz(dev, num_bits, NUM_LAYER)

#* guidance weight
LAMD = 0.5

#* run with time recording
ETA = 0.04 # step_size here
MAX_ITER = 30
record = []
t0 = time()
trainAndCompare(record, pqc, symmetry, train_data, test_data, ETA, LAMD, MAX_ITER)
t1 = time()
print(f"Run time: {(t1-t0)/60 : .2f} min")