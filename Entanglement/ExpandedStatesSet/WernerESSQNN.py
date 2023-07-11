import pennylane as qml
import pennylane.numpy as np

#! ------------------------ Ansatz 1 --------------------------------

# Define the quantum devices and QNodes
dev1, dev2 = qml.device("default.mixed", wires=2), qml.device("default.mixed", wires=2)

def enc_func(rho, parameters):
    qml.QubitDensityMatrix(rho, wires=[0,1])

def ansatz(parameters):
    # parameters = parameters[0]
    # print(parameters)
    qml.RY(parameters[0], wires=0)
    qml.RY(parameters[1], wires=1)
    # qml.CNOT(wires=[0, 1])
    # qml.RY(parameters[2], wires=0)
    # qml.RY(parameters[3], wires=1)
    # qml.CNOT(wires=[1, 0])
    # qml.RY(parameters[4], wires=0)
    # qml.RY(parameters[5], wires=1)

#! ------------------------ Paralled Enc --------------------------------

# Define the quantum devices and QNodes
dev1_p2, dev2_p2 = qml.device("default.mixed", wires=4), qml.device("default.mixed", wires=4)

def enc_func_p2(rho, parameters):
    qml.QubitDensityMatrix(rho, wires=[0,1])
    qml.QubitDensityMatrix(rho, wires=[2,3])

def enc_func_a2(rho, parameters):
    qml.QubitDensityMatrix(rho, wires=[0,1])
    # qml.BasisState(np.zeros(2), wires=[2,3])

def ansatz_p2(parameters):
    qml.RY(parameters[0], wires=0)
    qml.RY(parameters[1], wires=1)
    qml.RY(parameters[2], wires=2)
    qml.RY(parameters[3], wires=3)
    # qml.RX(parameters[0], wires=0)
    # qml.RX(parameters[1], wires=1)
    # qml.RX(parameters[2], wires=2)
    # qml.RX(parameters[3], wires=3)
    qml.CNOT(wires=[0, 1])
    # qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    # qml.CNOT(wires=[3, 0])
    qml.RY(parameters[4], wires=0)
    qml.RY(parameters[5], wires=1)
    qml.RY(parameters[6], wires=2)
    qml.RY(parameters[7], wires=3)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[3, 2])
    # qml.RX(parameters[8], wires=0)
    # qml.RX(parameters[9], wires=1)
    # qml.RX(parameters[10], wires=2)
    # qml.RX(parameters[11], wires=3)

def ansatz_p2_ps_c2(parameters):
    """expanded circuit form the minimum circuit: complexed 2"""
    # part3
    qml.U3(parameters[0], parameters[1], parameters[2], wires=0)
    qml.U3(parameters[3], parameters[4], parameters[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.U3(parameters[6], parameters[7], parameters[8], wires=0)
    qml.U3(parameters[9], parameters[10], parameters[11], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.U3(parameters[12], parameters[13], parameters[14], wires=0)
    qml.U3(parameters[15], parameters[16], parameters[17], wires=1)

    qml.U3(parameters[18], parameters[19], parameters[20], wires=2)
    qml.U3(parameters[21], parameters[22], parameters[23], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.U3(parameters[24], parameters[25], parameters[26], wires=2)
    qml.U3(parameters[27], parameters[28], parameters[29], wires=3)
    qml.CNOT(wires=[3, 2])
    qml.U3(parameters[30], parameters[31], parameters[32], wires=2)
    qml.U3(parameters[33], parameters[34], parameters[35], wires=3)

    # part2
    qml.ctrl(qml.PauliX, (0, 1, 2), control_values=(0, 1, 0))(wires=3)
    qml.ctrl(qml.PauliX, (0, 1, 2), control_values=(0, 1, 1))(wires=3)

    # part1
    qml.U3(parameters[36], parameters[37], parameters[38], wires=0)
    qml.U3(parameters[39], parameters[40], parameters[41], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.U3(parameters[42], parameters[43], parameters[44], wires=0)
    qml.U3(parameters[45], parameters[46], parameters[47], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.U3(parameters[48], parameters[49], parameters[50], wires=0)
    qml.U3(parameters[51], parameters[52], parameters[53], wires=1)

    qml.U3(parameters[54], parameters[55], parameters[56], wires=2)
    qml.U3(parameters[57], parameters[58], parameters[59], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.U3(parameters[60], parameters[61], parameters[62], wires=2)
    qml.U3(parameters[63], parameters[64], parameters[65], wires=3)
    qml.CNOT(wires=[3, 2])
    qml.U3(parameters[66], parameters[67], parameters[68], wires=2)
    qml.U3(parameters[69], parameters[70], parameters[71], wires=3)


def ansatz_p2_ps_c1(parameters):
    """expanded circuit form the minimum circuit: complexed 1"""
    # part3
    qml.U3(parameters[0], parameters[1], parameters[2], wires=0)
    qml.U3(parameters[3], parameters[4], parameters[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.U3(parameters[6], parameters[7], parameters[8], wires=0)
    qml.U3(parameters[9], parameters[10], parameters[11], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.U3(parameters[12], parameters[13], parameters[14], wires=0)
    qml.U3(parameters[15], parameters[16], parameters[17], wires=1)

    # part2
    qml.ctrl(qml.PauliX, (0, 1, 2), control_values=(0, 1, 0))(wires=3)
    qml.ctrl(qml.PauliX, (0, 1, 2), control_values=(0, 1, 1))(wires=3)

    # part1
    qml.U3(parameters[18], parameters[19], parameters[20], wires=2)
    qml.U3(parameters[21], parameters[22], parameters[23], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.U3(parameters[24], parameters[25], parameters[26], wires=2)
    qml.U3(parameters[27], parameters[28], parameters[29], wires=3)
    qml.CNOT(wires=[3, 2])
    qml.U3(parameters[30], parameters[31], parameters[32], wires=2)
    qml.U3(parameters[33], parameters[34], parameters[35], wires=3)

def ansatz_p2_ps_minimum(parameters):
    """caculated circuit form

    haved tested and works well
    """
    # part3
    qml.U3(parameters[0], parameters[1], parameters[2], wires=0)
    qml.U3(parameters[3], parameters[4], parameters[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.U3(parameters[6], parameters[7], parameters[8], wires=0)
    qml.U3(parameters[9], parameters[10], parameters[11], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.U3(parameters[12], parameters[13], parameters[14], wires=0)
    qml.U3(parameters[15], parameters[16], parameters[17], wires=1)

    # part2
    qml.ctrl(qml.PauliX, (0, 1, 2), control_values=(0, 1, 0))(wires=3)
    qml.ctrl(qml.PauliX, (0, 1, 2), control_values=(0, 1, 1))(wires=3)




