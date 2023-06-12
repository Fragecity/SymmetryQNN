import pennylane as qml
from Ansatzs.PQCBase import PQCBase


class CDThree(PQCBase):
    def __init__(self, dev, num_bits, num_layers):
        super().__init__(dev)
        self.num_bits = num_bits
        self.num_layers = num_layers

    def ansatz(self, params):
        '''PQC w/o input encoding'''
        qml.broadcast(qml.RX, wires=range(self.num_bits),
                      pattern="single", parameters=params[:3])
        qml.broadcast(qml.CRY, wires=range(3), pattern="chain", parameters=params[3:5])
        qml.broadcast(qml.RX, wires=range(self.num_bits),
                      pattern="single", parameters=params[6:])

    @property
    def U_circ(self):
        """return U_circ_real function when called"""

        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def U_circ_real(params):
            """turning the ansatz into a unitary matrix"""
            self.ansatz(params)
            return qml.expval(qml.Identity(0))

        return U_circ_real

    def circuit(self, x, params):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def circuit_real(x, params):
            """whole circuit"""
            # * data encoding
            qml.BasisEmbedding(x, wires=range(self.num_bits))

            # * ansatz
            self.ansatz(params)

            #* measurement (won't this conflict with the ob in SG?)
            return qml.expval(qml.PauliZ(self.num_bits - 1))

        return circuit_real(x, params)

    def draw_circ(self, params):
        """draw circuit in default mode, show directly"""
        fig, ax = qml.draw_mpl(self.U_circ)(params)
        fig.show()
