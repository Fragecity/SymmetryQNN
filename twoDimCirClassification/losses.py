import pennylane as qml
import pennylane.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

from Utils import hs_norm

sgRecord = {
    'cnt': 0,
    'valList': []
}

def qnnPrediction(qnn, params, x):
    predVal = qnn.circuit(x, params)
    return 1 if predVal>=0 else -1

def cost(params, qnn, data_batch):
    lossList = []
    for x, label in data_batch:
        predVal = qnn.circuit(x, params)
        lossList.append((predVal - label)**2)
    return np.mean(np.array(lossList))

def accuracy(qnn, params, data_batch):

    num_correct = 0
    for x, label in data_batch:
        prediction = qnnPrediction(qnn, params, x)
        num_correct += int(prediction == label)

    return num_correct / len(data_batch)

def costG(params, qnn, symmetry, lamd, data_batch, checkMode=False, returnG=False):

    cst = cost(params, qnn, data_batch)

    U = np.squeeze(qml.matrix(qnn.U_circ)(params))
    U = np.array(U._value) if isinstance(U, ArrayBox) else U
    g = symmetry.symmetry_guidance(U)

    if checkMode:
        # print(f"g: {g}")
        sgRecord['cnt'] += 1
        sgRecord['valList'].append(g)
        # g = (1-params[0])**2

    return g if returnG else cst + lamd * g

# from Utils import dagger, Z, tensorOfListOps2, SWAP

# def costG(params, qnn, symmetry, lamd, data_batch):
#     ob = tensorOfListOps2(Z, Z, Z, Z)
#     S2 = tensorOfListOps2(SWAP, SWAP)
#
#     U = np.squeeze(qml.matrix(qnn.U_circ)(params))
#     if isinstance(U, ArrayBox):
#         U = np.array(U._value)
#
#     O_tilde = dagger(U) @ ob @ U
#
#     # PO
#     Po2 = dagger(S2) @ O_tilde @ S2
#     PO = (O_tilde + Po2)/2
#
#     # HS norm
#     Diff = PO - O_tilde
#     HS_norm = hs_norm(Diff)
#     # print(f"HS norm: {HS_norm}")
#
#     return HS_norm