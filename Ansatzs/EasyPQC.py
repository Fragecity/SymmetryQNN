import pennylane as qml
from Ansatzs.PQCBase import PQCBase

#TODO: to be tested
class EasyPQC(PQCBase):
    def __init__(self, dev, enc_func, ansatz, ansatzOb, wCirOb):
        super().__init__(dev)
        self.enc_func = enc_func
        self.ansatz = ansatz
        self.ansatzOb = ansatzOb
        self.wCirOb = wCirOb

    @property
    def U_circ(self):
        """return U_circ_real function when called"""
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def U_circ_real(parameters):
            """turning the ansatz into a unitary matrix"""
            self.ansatz(parameters)
            return qml.expval(self.ansatzOb)

        return U_circ_real

    def circuit(self, x, parameters):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def circuit_real(x, parameters):
            """whole circuit"""
            self.enc_func(x, parameters)
            self.ansatz(parameters)
            return qml.expval(self.wCirOb)

        return circuit_real(x, parameters)

