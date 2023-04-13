import pennylane as qml
from Ansatzs.PQCBase import PQCBase

class YAnsatz(PQCBase):
    def __init__(self, dev, num_bits, num_layers):
        super().__init__(dev)
        self.num_bits = num_bits
        self.num_layers = num_layers

    def ansatz(self, params):
        '''PQC w/o input encoding'''
        qml.RY(params[0], wires=0)
        # qml.RY(params[1], wires=0)

    @property
    def U_circ(self):
        """return U_circ_real function when called"""
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def U_circ_real(parameters):
            """turning the ansatz into a unitary matrix"""
            self.ansatz(parameters)
            return qml.expval(qml.Identity(0))

        return U_circ_real
    
    def circuit(self, x, parameters):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def circuit_real(x, parameters):
            """whole circuit"""
            #* data encoding
            qml.AmplitudeEmbedding(x, wires=0, normalize=True)
            
            #* ansatz
            self.ansatz(parameters)
            return qml.expval(qml.PauliZ(self.num_bits - 1))

        return circuit_real(x, parameters)

