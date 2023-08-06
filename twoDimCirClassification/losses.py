import pennylane as qml
import pennylane.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

from Utils import hs_norm

def cost(qnn, params, data_batch):
    lossList = []
    for x, label in data_batch:
        predVal = qnn.circuit(x, params)
        lossList.append((predVal - label)**2)
    return np.mean(np.array(lossList))

def accuracy(qnn, params, data_batch):

    num_correct = 0
    for x, label in data_batch:
        predVal = qnn.circuit(x, params)
        prediction = 1 if predVal>=0 else -1
        num_correct += int(prediction == label)

    return num_correct / len(data_batch)

def costG(qnn, params, symmetry, lamd, data_batch):

    cst = cost(qnn, params, data_batch)

    U = np.squeeze(qml.matrix(qnn.U_circ)(params))

    if isinstance(U, ArrayBox):
        U = np.array(U._value)

    g = symmetry.symmetry_guidance(U)

    return cst + lamd * g