from twoDimCirClassification.QNNRepo.MidCircuit import MidCircuit
from QNNLib import cir1
import pennylane as qml


class type2MC(MidCircuit):
    def __init__(self, num_blocks):
        super().__init__()
        self.name = "cir2wl"
        self.num_params_each_block = 8
        self.num_blocks = num_blocks

    def circuit(self, params):
        for i in range(self.num_blocks):
            cir1(params[8 * i:8 * (i + 1)])
            qml.CNOT(wires=[3, 2])
            qml.CNOT(wires=[2, 1])
            qml.CNOT(wires=[1, 0])

    @property
    def num_params(self):
        return self.num_params_each_block * self.num_blocks

