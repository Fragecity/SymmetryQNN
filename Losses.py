import pennylane.numpy as np
import pennylane as qml


def cost(parameters, pqc, train_data):
    """cost function without symmetry guidance"""
    cst_values = []
    for x, label in train_data:
        cst_values.append(
            (pqc.circuit(x, parameters) + (-1) ** label) ** 2
            # abs(pqc.circuit(x, parameters) + (-1) ** label)
        )
    return sum(cst_values) / len(train_data)


def costG(parameters, pqc, symmetry, lamd, train_data):
    """cost function with symmetry guidance"""
    U = np.squeeze(qml.matrix(pqc.U_circ)(parameters))
    cst = cost(parameters, pqc, train_data)
    g = symmetry.symmetry_guidance(U)
    return cst + lamd * g


def accuracy(parameters, pqc, test_data):
    correct = 0
    for rho, label in test_data:
        prediction = np.ceil(pqc.circuit(rho, parameters))
        if np.abs(prediction - label) < 1: correct += 1
    return correct / len(test_data)

#TODO: why whole train_data passed rather than parched ones?
