"""Store some expanded Tools, which are not used frequently."""

import numpy as np
from Utils import X, Y, Z, Id

#! ---------------------- 2-qubit Pauli Matrices ---------------------- #

II = np.kron(Id, Id)
IX = np.kron(Id, X)
IY = np.kron(Id, Y)
IZ = np.kron(Id, Z)
XI = np.kron(X, Id)
XX = np.kron(X, X)
XY = np.kron(X, Y)
XZ = np.kron(X, Z)
YI = np.kron(Y, Id)
YX = np.kron(Y, X)
YY = np.kron(Y, Y)
YZ = np.kron(Y, Z)
ZI = np.kron(Z, Id)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)
ZZ = np.kron(Z, Z)

