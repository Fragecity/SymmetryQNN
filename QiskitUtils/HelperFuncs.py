import numpy as np
from qiskit.quantum_info.operators import Operator

def getUofCircuit(paras, qc):
    value_dict = dict(zip(qc.parameters, paras))
    U = np.matrix(Operator(qc.bind_parameters(value_dict)))
    return U