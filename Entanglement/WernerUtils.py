from itertools import product
import pennylane as qml
import numpy as np
from Utils import X, Y, Z, Id, SWAPij
from functools import reduce
from operator import matmul
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator

#! --------------------------------- Basic operations --------------------------------- #

def pauli(i):
    if i == 0: return Id
    elif i == 1: return X
    elif i == 2: return Y
    elif i == 3: return Z

def tensor(*U):
    U_res = 1
    if len(U)>0:
        for u in U: U_res = np.kron(np.array(u), U_res)
        return U_res
    return None

def pauli_group(n):
    plus = [tensor(*list(map(pauli, index)))
            for index in product([0,1,2,3], repeat = n) ]
    minus = [-tensor(*list(map(pauli, index)))
            for index in product([0,1,2,3], repeat = n) ]
    return plus + minus

def swap_2sys(num_qubit):
    """SwapOperator that swaps each qubit of the first subsystem with the corresponding qubit of the second subsystem"""
    dev = qml.device('default.mixed', wires=2 * num_qubit)

    @qml.qnode(dev)
    def swap_circ():
        for i in range(num_qubit):
            qml.SWAP([i, i + num_qubit])
        return qml.expval(qml.Identity(wires=range(2 * num_qubit)))

    return qml.matrix(swap_circ)()

def bell_unitary(num_qubit):
    dev = qml.device('default.mixed', wires=2*num_qubit)
    @ qml.qnode(dev)
    def bell_circ():
        for i in range(num_qubit):
            qml.CNOT([i, i+ num_qubit])
            qml.Hadamard(i)
        return qml.expval(qml.PauliZ(0))
    return qml.matrix(bell_circ)()


#! --------------------------------- Werner state_Qk --------------------------------- #

def wernerWithLabel(num_qubit):
    """Generate a single Werner state with label"""
    d = 2**num_qubit
    f = np.random.random()*2 - 1
    swp_lst = [ SWAPij(i,j,2*num_qubit) for i,j in zip(range(num_qubit), range(num_qubit, 2*num_qubit))]
    swap = reduce(matmul, swp_lst)
    werner_state = 1/(d**3 - d) * (
        (d - f)*np.identity(d**2) + (d*f - 1)*swap ) 
    return (werner_state, 0<=f<=1)

def wernerState(num_qubit, f):
    d = 2**num_qubit
    # f = np.random.random()*2 - 1
    swp_lst = [ SWAPij(i,j,2*num_qubit) for i,j in zip(range(num_qubit), range(num_qubit, 2*num_qubit))]
    swap = reduce(matmul, swp_lst)
    werner_state = 1/(d**3 - d) * (
        (d - f)*np.identity(d**2) + (d*f - 1)*swap ) 
    return werner_state

def genWernerDataWithLabel(num_part, num_data):
    """
    Generate a list of Werner state with label
    label : f>0 -> 1, f<0 -> -1"""
    data = []
    for _ in range(num_data):
        f = 2*np.random.random() - 1
        label = 1 if f>=0 else -1
        data.append((wernerState(num_part, f), label))
    return data

def witness_Qk(num_qubit):
    qc = QuantumCircuit( 2*num_qubit)
    qc.swap(range(num_qubit), range(num_qubit, 2*num_qubit))
    return np.array(Operator(qc))


#! --------------------------------- Werner state_PL --------------------------------- #

def swap_reg(register1, register2):
    for i,j in zip(register1, register2): qml.SWAP([i,j])

def swap_gate(num_qubit):
    dev = qml.device("default.mixed", wires=2*num_qubit)
    @qml.qnode(dev, diff_method="backprop", interface="autograd")
    def circuit(num_part):
        swap_reg(range(num_part), range(num_part, 2*num_part))
        return qml.expval(qml.PauliZ(0))
    return qml.matrix(circuit)(num_qubit)

#! this seems replicate functions above
def swap_2sys(num_qubit):
    dev = qml.device('default.mixed', wires=2*num_qubit)
    
    @ qml.qnode(dev)
    def swap_circ():
        for i in range(num_qubit):
            qml.SWAP([i, i+num_qubit])
        return qml.expval(qml.Identity(wires=range(2*num_qubit )))
    
    return qml.matrix(swap_circ)()

def wernerState_PL(num_qubit, f):
    d = 2**num_qubit
    # swap = swap_gate(num_qubit)
    swap = swap_2sys(num_qubit)
    werner_state = 1/(d**3 - d) * (
                (d - f)*np.identity(d**2) + (d*f - 1)*swap) 
    return np.array(werner_state, requires_grad = False)

def genWernerDataWithLabel_PL(num_part, num_data):
    """label : f>0 -> 1, f<0 -> -1"""
    data = []
    for _ in range(num_data):
        f = 2*np.random.random() - 1
        label = 1 if f>=0 else -1
        data.append((wernerState_PL(num_part, f), label))
    return data

def witness_PL(num_qubit):
    return swap_gate(num_qubit)

#! --------------------------------- 2-qubit entanglement separation --------------------------------- #

#* PPT criterion for 2-qubits system
def partial_transpose(rho, dims):
    """
    Inputs:
        rho: 输入的密度矩阵
        dims: 系统的维度，对于一个二比特系统，dims=[2,2]
    """ 
    n = len(dims)
    transposed_rho = rho.copy()
    for i in range(n):
        for j in range(n):
            #* fetch block
            block = rho[i*dims[0]:(i+1)*dims[0], j*dims[1]:(j+1)*dims[1]]
            #* replace with transposed block
            transposed_rho[i*dims[0]:(i+1)*dims[0], j*dims[1]:(j+1)*dims[1]] = block.transpose()
    return transposed_rho

def is_positive_semi_definite(matrix):
    """检查矩阵的所有本征值是否都非负(sympy)"""
    return all(eigenval >= 0 for eigenval in matrix.eigenvals().keys())

def is_entangled(rho, dims):
    """all three are used for checking sympy matrix"""
    # 计算偏置转置
    rho_pt = partial_transpose(rho, dims)
    # 检查偏置转置后的矩阵是否半正定
    return not is_positive_semi_definite(rho_pt)