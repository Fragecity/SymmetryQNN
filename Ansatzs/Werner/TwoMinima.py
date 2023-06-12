import pennylane as qml
from Ansatzs.PQCBase import PQCBase


class TwoMinima(PQCBase):
    def __init__(self, dev, Hermi_Ob):
        super().__init__(dev)
        self.dev = dev
        self.Hermi_Ob = Hermi_Ob

    def layer(self, parameters, pattern):
        """layer-build helper function for ansatz construction"""
        qml.broadcast(qml.RY, wires=range(self.num_bits),
                      pattern="single", parameters=parameters)
        qml.broadcast(qml.CNOT, wires=range(self.num_bits), pattern=pattern)

    def ansatz(self, parameters):
        """PQC w/o input encoding"""
        # qml.QubitUnitary(reflection(para[0]), 0)
        # qml.QubitUnitary(reflection(para[1]), 1)

        # qml.U3(para[0], para[1], para[2], 0)
        # qml.U3(para[3], para[4], para[5], 1)
        qml.CNOT([0, 1])
        qml.U3(parameters[6], parameters[7], parameters[8], 0)
        # qml.U3(para[9], para[10], para[11], 1)
        # qml.QubitUnitary(reflection(para[12]), 0)
        # qml.QubitUnitary(reflection(para[13]), 1)

    @property
    def U_circ(self):
        """return U_circ_real function when called"""

        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def U_circ_real(parameters):
            """turning the ansatz into a unitary matrix"""
            self.ansatz(parameters)
            return qml.expval(qml.Hermitian(self.Hermi_Ob, wires=[0, 1]))

        return U_circ_real

    def circuit(self, x, parameters):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def circuit_real(x, parameters):
            """whole circuit, x is a density matrix here"""
            qml.QubitDensityMatrix(x, wires=[0, 1])
            self.ansatz(parameters)
            return qml.expval(qml.Hermitian(self.Hermi_Ob, wires=[0, 1]))

        return circuit_real(x, parameters)
