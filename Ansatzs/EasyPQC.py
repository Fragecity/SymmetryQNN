import pennylane as qml
from Ansatzs.PQCBase import PQCBase


class EasyPQC(PQCBase):
    def __init__(self, dev, enc_func, ansatz, ansatzOb, wCirOb):
        super().__init__(dev)
        self.enc_func = enc_func
        self.ansatz = ansatz
        self.ansatzOb = ansatzOb
        self.wCirOb = wCirOb

    def getEncodedDM(self, x, parameters):
        """return encoded density matrix when called"""

        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def encodedDM(x, parameters):
            """return encoded density matrix"""
            self.enc_func(x, parameters)
            return qml.density_matrix(range(self.dev.num_wires))

        return encodedDM(x, parameters)

    @property
    def U_circ(self):
        """return U_circ_real function when called"""

        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def U_circ_real(parameters):
            """turning the ansatz into a unitary matrix, not including the ob?
            Note: real here means the true function, not about complex numbers.
            """
            self.ansatz(parameters)
            return qml.expval(self.ansatzOb)

        return U_circ_real

    def circuit(self, x, parameters):
        """return circuit_real function, which evals the whole circuit using wCirOb, when called"""

        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def circuit_real(x, parameters):
            """whole circuit
            Note: real here means the true function, not about complex numbers.
            """
            self.enc_func(x, parameters)
            self.ansatz(parameters)
            return qml.expval(self.wCirOb)

        return circuit_real(x, parameters)

    # def getEncodedState(self, x, parameters):
    #     """return encoded state when called"""
    #     @qml.qnode(self.dev, diff_method="backprop", interface="torch")
    #     def encodedState(x, parameters):
    #         """encoded state"""
    #         self.enc_func(x, parameters)
    #         return qml.state()

    #     return encodedState(x, parameters)

