import copy
import numpy as np
from numpy import kron as kr
from numpy import pi, sin, cos, exp, trace
from functools import reduce
# basic gates
C0 = np.array([[1,0], [0,0]])
C1 = np.array([[0,0], [0,1]])
X = np.array([[0,1], [1,0]])
Z = np.array([[1,0], [0,-1]])
Id = np.array([[1,0], [0,1]])


#%% Groups operator

def generate_freegroup(a_set):
    FreeGroup = list()
    gates_queq = copy.deepcopy(a_set)

    while gates_queq:
        gate = gates_queq.pop()
        if not mat_in_set(gate, FreeGroup):
            FreeGroup.append(gate)
            for s in a_set: gates_queq.append(gate @ s)
    
    return FreeGroup

def order(mat) -> int:
    power = mat.copy() 
    order = 1
    while not equal(power, np.identity(len(mat))): 
        power = power @ mat
        order += 1
    return order


def Twirling(Group: list, operator) -> np.array:
    lst = [G @ operator @ dagger(G) for G in Group]
    return 1/len(Group) * sum(lst) 

def Ad(G, g) -> np.array:
    "Ad_G(g) = GgG^\dagger"
    return G @ g @ dagger(G)

#%% gates algebra 
def expend_U(U: np.array, i, q_num) -> np.array:
     Idl, Idr = 1,1
     for k in range(i): Idl = np.kron(Idl, Id)
     for k in range(q_num-i-1): Idr = np.kron(Idr, Id)
     return kr(kr(Idl, U), Idr)

def convert_para_2_multi_gates(U: callable,paras:list) -> np.array:
    U_res = 1
    if len(paras)>0:
        for i in range(len(paras)): U_res = np.kron(U_res, U(paras[i]))
        return U_res
    return None


def equal(A,B) -> bool:
    B = np.abs(A-B) < 0.01
    return B.all()

def dagger(A) -> np.matrix:
    return np.conj(np.matrix(A).transpose())


def mat_in_set(A, Group) -> bool:
     for G in Group:
          if equal(A, G): return True
     return False

def tensor(*U):
    U_res = 1
    if len(U)>0:
        for u in U: U_res = np.kron(U_res, np.array(u))
        return U_res
    return None

# %% Some special gates
def CX(i,j, q_num) -> np.array:
     Idl, Idm, Idr = 1,1,1
     for k in range(min(i,j)): Idl = np.kron(Idl, Id)
     for k in range(np.abs(i-j)-1): Idm = np.kron(Idm, Id)
     for k in range(q_num-max(i,j)-1): Idr = np.kron(Idr, Id)
     if i<j:
          CNOT = kr(kr(kr( kr(Idl, C0), Idm), Id), Idr) \
               + kr(kr(kr( kr(Idl, C1), Idm), X), Idr)
     elif i>j:
          CNOT = kr(kr(kr( kr(Idl, Id), Idm), C0), Idr) \
               + kr(kr(kr( kr(Idl, X), Idm), C1), Idr) 
     else:
          return None
     return CNOT

def SWAP(i,j, q_num) -> np.array:
    if i == j:
        return None
    down = CX(i,j, q_num)
    up = CX(j,i, q_num)
    return up @ down @ up

def Ry(theta) -> np.array:
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

def Phase(phi) -> np.array:
    res = np.array([
        [1,0],
        [0,exp(1j * phi)]
    ])
    return res
#%% Quantum Neural Net

class QNN_naive():
    O = []
    def set_paras(self, parameters:list)-> None:
        self.qbits_number = len(parameters)//2
        self.left_paras = copy.deepcopy(parameters[:self.qbits_number])
        self.right_paras = copy.deepcopy(parameters[self.qbits_number:])
    
    def set_observable(self, O: np.array) -> None:
        self.O = copy.deepcopy(O)
        self.cost = lambda A: trace()

    def set_O(self , O) -> None:
        if len(O) != 0:
            self.O = O
                    

    def initialize(self, state) -> None:
        if len(state) != 0:
            state = np.array(state)
            if len(state.shape) == 1:
                self.state = dagger(state) @ np.matrix(state)
            else:
                self.state = state
        else:
            state = np.zeros(2**self.qbits_number)
            state[0] = 1
            self.state = state

    def __init__(self, qbits_number, O = [], init_state = []) -> None:
        self.qbits_number = qbits_number
        # Now, construct the neural net
        if qbits_number < 3:
            self.CXs = CX(0,1,2)
        else:
            self.CXs = reduce(np.multiply, [CX(i, (i+1) % qbits_number, qbits_number) for i in range(qbits_number) ] )
        
        self.initialize(init_state)
        self.set_O(O)
        
    def __call__(self, *parameters: list) -> np.array:
        if parameters:
            self.set_paras(parameters[0])
        Rys_left = convert_para_2_multi_gates(Ry, self.left_paras)
        Rys_right = convert_para_2_multi_gates(Ry, self.right_paras)
        
        return Rys_left @ self.CXs @ Rys_right

    def f_hat(self) -> float:
        result = self() @ self.state @ dagger(self()) 
        return trace(result @ self.O)


# %%
# if __name__ == "main":
# O = kr(Id, Z)
# QNN = QNN_naive(2,O= O)
# QNN([0,0,0,0])
# QNN.cost()
# %%
