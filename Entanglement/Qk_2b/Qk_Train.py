import pickle
import numpy as np
from Utils import Z
from Utils import tensorOfListOps as tensor
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from QiskitUtils.MyQC import QNN
from QiskitUtils.TrainTool import grad_des, trainTwo
from QiskitUtils.Losses import squareLoss

from functools import partial

#* dataRoot
data_path = './data/test_2b_dns_1.pkl'
save_path = './data/test_2b_dns_1_result1.pkl'

#* Training settings
MAX_ITER = 2 # 40
ETA = 0.02

LAMBDA = 4

#* cost function is defined in the Training part

#! ----------------------------- DNS Setting ----------------------------- #

#* dns load
with open(data_path, 'rb') as f:
    [qnn_s, qnn_t, symmetry] = pickle.load(f)

#? why a new qnn
NUM_PART = 1
qc = QuantumCircuit(NUM_PART*6,1)
rand_unitary = RealAmplitudes(NUM_PART*4, reps=2,entanglement='linear').decompose()
qc.append(rand_unitary, range(2*NUM_PART, 6*NUM_PART))


#* Symmetry
observable = tensor(np.identity(2**(5*NUM_PART)), Z)
symmetry.observable = observable

#! ----------------------------- Training -------------------------------- #

cost = partial(squareLoss, qnn=qnn_s)

para_list, func_list,  paraG_list, funcG_list = trainTwo(cost, qc, qnn_s, symmetry, MAX_ITER, ETA, LAMBDA)

#! ----------------------------- Save and Plot ----------------------------- #

with open(save_path, 'wb') as f:
    pickle.dump([para_list, func_list,  paraG_list, funcG_list], f)

import matplotlib.pyplot as plt
# %matplotlib
plt.plot(func_list, label='without guidance')
plt.plot(funcG_list, label='with guidance')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('costs')
plt.title('landscape during training')
plt.draw()
plt.show()

