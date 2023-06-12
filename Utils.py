import numpy as np
import os
from itertools import product
from numpy import kron as kr
from functools import reduce
from operator import add
from collections import Iterable
#TODO: separate below into several files?

# * dataRootPath
project_root = os.path.abspath(os.path.dirname(__file__))
dataRepo_path = os.path.join(project_root, 'DataRepo')

#! -------------------------Basic Gates-------------------------
PauliName_dict = {0:'I', 1:'X', 2:'Y', 3:'Z'}

X = np.array([[0, 1],
              [1, 0]])
Y = np.array([[0, -1j],
              [1j, 0]])
Z = np.array([[1, 0],
              [0, -1]])
Id = np.array([[1, 0],
               [0, 1]])

def pauli(i):
    """fetch Pauli basic ops in numeric order: Id, X, Y, Z"""
    if i == 0: return Id
    elif i == 1: return X
    elif i == 2: return Y
    elif i == 3: return Z

HADAMARD = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                   [1/np.sqrt(2), -1/np.sqrt(2)]])

C0 = np.array([[1,0], 
               [0,0]])
C1 = np.array([[0,0], 
               [0,1]])

def Ry(theta) -> np.array:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def Phase(phi) -> np.array:
    res = np.array([
        [1,0],
        [0,np.exp(1j * phi)]
    ])
    return res

SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

def CXij(i,j, q_num) -> np.array:
    Idl, Idm, Idr = identity_qb(min(i,j)), identity_qb(np.abs(i-j)-1), identity_qb(q_num-max(i,j)-1)
    if i<j:
        return tensorOfListOps(Idl, C0, Idm, Id, Idr) + tensorOfListOps(Idl, C1, Idm, X, Idr)
    elif i>j:
        return tensorOfListOps(Idl, Id, Idm, C0, Idr) + tensorOfListOps(Idl, X, Idm, C1, Idr)
    else: return None

# def CXij(i,j, num_qbits) -> np.array:
#      Idl, Idm, Idr = 1,1,1
#      for k in range(min(i,j)): Idl = np.kron(Idl, Id)
#      for k in range(np.abs(i-j)-1): Idm = np.kron(Idm, Id)
#      for k in range(num_qbits-max(i,j)-1): Idr = np.kron(Idr, Id)
#      if i<j:
#           CNOT = kr(kr(kr( kr(Idl, C0), Idm), Id), Idr) \
#                + kr(kr(kr( kr(Idl, C1), Idm), X), Idr)
#      elif i>j:
#           CNOT = kr(kr(kr( kr(Idl, Id), Idm), C0), Idr) \
#                + kr(kr(kr( kr(Idl, X), Idm), C1), Idr) 
#      else:
#           return None
#      return CNOT

def SWAPij(i,j, num_qbits) -> np.array:
    if i == j:
        return None
    down = CXij(i,j, num_qbits)
    up = CXij(j,i, num_qbits)
    return up @ down @ up

# and this name should be changed
# def SWAP(i,j, q_num)  -> np.array:
#     if isinstance(i, int):
#         return SWAPij(i,j, q_num)
#     elif isinstance(i, Iterable):
#         for i_item, j_item in zip(i,j):
#             pass

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

CNOT2 = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]])

def reflection(theta):
    """Reflection Operator"""
    return np.array([[np.cos(theta), np.sin(theta)],
                     [np.sin(theta), -np.cos(theta)]])

def identity_qb(num_qbits):
    if num_qbits <= 0: return 1
    else: return np.identity(2**num_qbits)

def pauli_group(n):
    """n means n qubits dim"""
    plus = [tensorOfListOps(*list(map(pauli, index))) 
            for index in product([0,1,2,3], repeat = n) ]
    minus = [-tensorOfListOps(*list(map(pauli, index))) 
            for index in product([0,1,2,3], repeat = n) ]
    return plus + minus

#! -------------------------Basic operations-------------------------
def dagger(A) -> np.array:
    A_conj = np.conj(np.array(A))
    return A_conj.T


def Adjoint(U: np.array, O: np.array) -> np.array:
    '''DM Transformation: return UOU^dagger'''
    return U @ O @ dagger(U)

# this tensor is consisting  with qiskit (KM)
def tensorOfListOps(*U):
    U_res = 1
    if len(U)>0:
        for u in U: 
            U_res = np.kron(np.array(u), U_res)
        return U_res
    return None

def hs_norm(A):
    """Hilbert Schmidt norm"""
    return np.real(np.trace(dagger(A) @ A))

def purify(rho):
    n = len(rho)
    Id = np.identity(n)
    rho = np.array(rho)
    eigvals, eigvecs = np.linalg.eig(rho)
    lin_comb_of_vecs =  [
        tensorOfListOps(Id[:, i], eigvecs[:,i] * np.sqrt(eigvals[i])) 
        for i in range(n)
    ]
    return reduce(add, lin_comb_of_vecs)

#! -------------------------Basic States-------------------------
def bellState(index:str):
    """
    Generate a Bell state in np.array based on the given index.
    """
    # Choose the corresponding Bell state based on the input index
    if index   == "psi_plus":
        return np.array([1,  0,  0,  1]) / np.sqrt(2)
    elif index == "psi_minus":
        return np.array([1,  0,  0, -1]) / np.sqrt(2)
    elif index == "phi_plus":
        return np.array([0,  1,  1,  0]) / np.sqrt(2)
    elif index == "phi_minus":
        return np.array([0,  1, -1,  0]) / np.sqrt(2)
    else:
        raise ValueError("Input index out of range")

def normalized(v: np.array) -> np.array:
    return v/np.linalg.norm(v)

def rand_pureState(num_qbits):
    dim = 2** num_qbits
    vec_real = np.random.random(dim)
    vec_img = np.random.random(dim) * 1j
    vec = vec_real + vec_img
    return normalized(vec)

def state2density(psi):
    psi = np.matrix(psi)
    return dagger(psi) @ psi

#! -------------------------Training-------------------------
def rand_paras(num_paras):
    """Gen rand parameters in range [0, 2pi] for given num_paras"""
    return np.random.random(num_paras) *2*np.pi

#! -------------------------Others-------------------------
def equal_tol(A, B, tol=1e-8):
    """check if two matrices are equal within tolerance, default abs_diff < 1e-8"""
    difference = np.matrix(A) - np.matrix(B)
    TF_mat = np.abs(difference) < tol
    return np.all(TF_mat)