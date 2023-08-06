import pennylane as qml
import pennylane.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

from Utils import hs_norm
from Entanglement.WernerUtils import wernerAncillaTwirl, wernerTwirling

def cost(model, params, data_batch):
    lossList = []
    for rho, label in data_batch:
        prediction = model(params, rho)
        lossList.append((prediction - label)**2)
    return np.mean(np.array(lossList))
    # rs = np.mean(np.array(lossList))
    # print(rs)
    # return rs
def accuracy(qnn1, qnn2, params, data_batch):

    num_correct = 0
    for rho, label in data_batch:
        tr1 = qnn1.circuit(rho, params[0])
        tr2 = qnn2.circuit(rho, params[1])
        prediction = 1 if (tr1 >= 0 and tr2 >= 0) else 0
        num_correct += int(prediction == label)

    return num_correct / len(data_batch)

def costG(qnn1, qnn2, model, params, num_each_substm, num_ancilla, lamd, data_batch):

    cst = cost(model, params, data_batch)

    U1 = np.squeeze(qml.matrix(qnn1.U_circ)(params[0]))
    U2 = np.squeeze(qml.matrix(qnn2.U_circ)(params[1]))
    if isinstance(U1, ArrayBox):
        U1 = np.array(U1._value)
        U2 = np.array(U2._value)

    if num_ancilla == 0:
        PO1 = wernerTwirling(num_each_substm, U1)
        PO2 = wernerTwirling(num_each_substm, U2)
    else:
        PO1 = wernerAncillaTwirl(num_each_substm, num_ancilla, U1)
        PO2 = wernerAncillaTwirl(num_each_substm, num_ancilla, U2)
    
    g = hs_norm(U1 - PO1) + hs_norm(U2 - PO2)

    return cst + lamd * g