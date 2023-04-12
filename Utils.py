import pennylane.numpy as np
import os


def dagger(A) -> np.array:
    A_conj = np.conj(np.array(A))
    return A_conj.T


def Adjoint(U: np.array, O: np.array) -> np.array:
    '''DM Transformation: return UOU^dagger'''
    return U @ O @ dagger(U)


def hs_norm(A):
    """Hilbert Schmidt norm"""
    return np.real(np.trace(dagger(A) @ A))


project_root = os.path.abspath(os.path.dirname(__file__))

dataRepo_path = os.path.join(project_root, 'DataRepo')
