import numpy as np
from functools import reduce
from operator import matmul
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator

from Symmetries.EasySymmetry import EasySymmetry
from Symmetries.HelperFuncs import generate_freegroup
from Utils import SWAPij, pauli_group, purify
from Utils import tensorOfListOps as tensor
from Entanglement.WernerUtils import wernerWithLabel
from QiskitUtils.MyQC import QNN

#* Data Generation
NUM_PART = 1
NUM_DATA = 64

e1 = np.zeros(2**(NUM_PART*2))
e1[0] = 1 #e1 is the 1000... (on the whole system) pure basis state
datas = [wernerWithLabel(NUM_PART) for _ in range(NUM_DATA) ]
pured_data = [ (tensor(purify(data), e1), label) for data, label in datas ]

CUT = int(0.8 * NUM_DATA)
S = pured_data[:CUT]
T = pured_data[CUT:]

#* PQC and its ini with data
qc = QuantumCircuit(NUM_PART*8 + 1, 1)
rand_unitary = RealAmplitudes(NUM_PART*4, reps=2,entanglement='linear').decompose()
qc.append(rand_unitary, range(4*NUM_PART, 8*NUM_PART))
qc.h(-1)
qc.cswap(-1, range(4*NUM_PART), range(4*NUM_PART, 8*NUM_PART))
qc.measure(-1,0)
qc = transpile(qc, AerSimulator())
qnn_s = QNN(qc)
qnn_t = QNN(qc)
qnn_s.encode('initialize', S)
qnn_t.encode('initialize', T)

#* Symmetry
swp_lst = [ SWAPij(i,j,2*NUM_PART) for i,j in zip(range(NUM_PART), 
                                                range(NUM_PART, 2*NUM_PART))  ]
swap = reduce(matmul, swp_lst)
paulis = pauli_group(NUM_PART)

IDd = np.identity(2**(2*NUM_PART))
sym_set = [tensor(pauli, pauli) for pauli in paulis]
sym_set.append(swap)
sym_set.append(swap)
sym_set = [tensor(IDd, op, IDd, IDd) for op in sym_set]
sym_group = generate_freegroup(sym_set)

symmetry = EasySymmetry(sym_group=sym_group)

#* Save
import pickle
with open('./Werner/data/one_qb_data.pkl', 'wb') as f:
    pickle.dump([qnn_s, qnn_t, symmetry],f)



if __name__ == '__main__':
    
    from qiskit.quantum_info import partial_trace
    import matplotlib.pyplot as plt 
    para = qnn_s.rand_paras()
    # qc = qnn_s.batch[0]
    # assigned_para = dict(zip(qc.parameters, para))
    # qcc = qc.bind_parameters(assigned_para)
    # backend = AerSimulator()
    # cnts = backend.run(qcc).result().get_counts()
    # print(cnts)
    print(qnn_s.run_batch(para))
    # qc.initialize(pured_data[0][0])
    # qc.measure(3*NUM_PART, 0)
    # cnt = backend.run(qc).result().get_counts()

    # qc.draw('mpl')
    # plt.show()
    # print(cnt )

    