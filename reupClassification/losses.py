import pennylane as qml
import pennylane.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

import sys

sys.path.append("./")
from Utils import hs_norm

sgRecord = {"cnt": 0, "valList": []}

# def qnnPrediction(qnn, params, x):
#     predVal = qnn.circuit(x, params)
#     return 1 if predVal>=0 else -1


def StepFunction(x):
    return 1 if x >= 0 else 0

def qnnPrediction(qnn_ob1, qnn_ob2, params, x):
    predVal1 = qnn_ob1.circuit(x, params)
    predVal2 = qnn_ob2.circuit(x, params)

    return StepFunction(predVal1) + StepFunction(predVal2)


# def cost(params, qnn, data_batch):
#     lossList = []
#     for x, label in data_batch:
#         predVal = qnn.circuit(x, params)
#         lossList.append((predVal - label)**2)
#     return np.mean(np.array(lossList))


def signoid(x):
    return 1 / (1 + np.exp(-6 * x))


def loss_threeClasses(predVal1, predVal2, label):
    pred_label1 = signoid(predVal1)
    pred_label2 = signoid(predVal2)
    return (pred_label1 + pred_label2 - label) ** 2


def cost(params, qnn_ob1, qnn_ob2, data_batch):
    lossList = []
    for x, label in data_batch:
        predVal1 = qnn_ob1.circuit(x, params)
        predVal2 = qnn_ob2.circuit(x, params)
        lossList.append(loss_threeClasses(predVal1, predVal2, label))
    return np.mean(np.array(lossList))


def costTwiredLoss(params, qnn_ob1, qnn_ob2, symmetry_ob1, symmetry_ob2, data_batch):

    U1 = np.squeeze(qml.matrix(qnn_ob1.U_circ)(params))
    U1 = np.array(U1._value) if isinstance(U1, ArrayBox) else U1
    O1_tilde, PO1 = symmetry_ob1._get_O_PO(U1)

    U2 = np.squeeze(qml.matrix(qnn_ob2.U_circ)(params))
    U2 = np.array(U2._value) if isinstance(U2, ArrayBox) else U2
    O2_tilde, PO2 = symmetry_ob2._get_O_PO(U2)

    lossList = []
    for x, label in data_batch:
        predVal = np.real(
            np.trace(qnn_ob1.getEncodedDM(x, params) @ PO1)
        )  # + qnn.circuit(x, params))
        predVal2 = np.real(np.trace(qnn_ob2.getEncodedDM(x, params) @ PO2))
        lossList.append(loss_threeClasses(predVal, predVal2, label))

    return np.mean(np.array(lossList))


# def accuracy(qnn, params, data_batch):
#
#     num_correct = 0
#     for x, label in data_batch:
#         prediction = qnnPrediction(qnn, params, x)
#         num_correct += int(prediction == label)
#
#     return num_correct / len(data_batch)


def accuracy(qnn_ob1, qnn_ob2, params, data_batch):

    num_correct = 0
    for x, label in data_batch:
        prediction = qnnPrediction(qnn_ob1, qnn_ob2, params, x)
        num_correct += int(prediction == label)

    return num_correct / len(data_batch)


def costG(params, qnn, symmetry, lamd, data_batch, checkMode=False, returnG=False):

    if not returnG:
        cst = cost(params, qnn, data_batch)

    U = np.squeeze(qml.matrix(qnn.U_circ)(params))
    U = np.array(U._value) if isinstance(U, ArrayBox) else U
    g = symmetry.symmetry_guidance(U)

    if checkMode:
        # print(f"g: {g}")
        sgRecord["cnt"] += 1
        sgRecord["valList"].append(g)
        # g = (1-params[0])**2

    if returnG:
        return g
    else:
        return cst + lamd * g


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
