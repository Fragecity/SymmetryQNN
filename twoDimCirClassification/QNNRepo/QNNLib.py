import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.mixed", wires=4)

def enc_func(x, parameters):
    """Encode classical data into quantum states."""
    #TODO: seems that this function could using args* instead of parameters
    qml.RX(x[0]*np.pi*2, wires=0)
    qml.RX(x[1]*np.pi*2, wires=1)
    qml.RX(x[0]*np.pi*2, wires=2)
    qml.RX(x[1]*np.pi*2, wires=3)
    
    qml.RY(np.pi/4, wires=0)
    qml.RY(np.pi/4, wires=1)
    qml.RY(np.pi/4, wires=2)
    qml.RY(np.pi/4, wires=3)

    qml.RZ(np.pi/4, wires=0)
    qml.RZ(np.pi/4, wires=1)
    qml.RZ(np.pi/4, wires=2)
    qml.RZ(np.pi/4, wires=3)

def cir_testSG(parameters):
    qml.RX(parameters[0], wires=0)
    qml.CZ(wires=[0,1])
    qml.RX(parameters[1], wires=1)

def cir1(parameters):
    qml.RX(parameters[0], wires=0)
    qml.RX(parameters[1], wires=1)
    qml.RX(parameters[2], wires=2)
    qml.RX(parameters[3], wires=3)

    qml.RZ(parameters[4], wires=0)
    qml.RZ(parameters[5], wires=1)
    qml.RZ(parameters[6], wires=2)
    qml.RZ(parameters[7], wires=3)

def cir2(parameters):
    cir1(parameters[:8])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[1,0])

def cir2wl(parameters, num_layers):
    for i in range(num_layers):
        cir1(parameters[8*i:8*(i+1)])
        qml.CNOT(wires=[3,2])
        qml.CNOT(wires=[2,1])
        qml.CNOT(wires=[1,0])

def cir2wl_cz(parameters, num_layers):
    for i in range(num_layers):
        cir1(parameters[8*i:8*(i+1)])
        qml.CZ(wires=[3,2])
        qml.CZ(wires=[2,1])
        qml.CZ(wires=[1,0])

def cir3(parameters):
    cir1(parameters[:8])
    qml.CRZ(parameters[8], wires=[3,2])
    qml.CRZ(parameters[9], wires=[2,1])
    qml.CRZ(parameters[10], wires=[1,0])