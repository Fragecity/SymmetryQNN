from reupClassification.QNNRepo.MidCircuit import MidCircuit
from reupClassification.QNNRepo.QNNLib import cir1, cir1_multi
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


class type2MC_multiEven(MidCircuit):
    def __init__(self, num_blocks, num_qubits):
        super().__init__()
        self.name = "cir2wl_multiEven"
        self.num_blocks = num_blocks
        self.num_qubits = num_qubits
        self.num_params_each_block = 2 * num_qubits

    def circuit(self, params):
        for i in range(self.num_blocks):
            cir1_multi(params[self.num_params_each_block * i:self.num_params_each_block * (i + 1)], self.num_qubits)
            parity = 1 if i % 2 == 1 else 0
            for j in range(parity, self.num_qubits-1, 2):
                qml.CNOT(wires=[j, j+1])

    @property
    def num_params(self):
        return self.num_params_each_block * self.num_blocks

