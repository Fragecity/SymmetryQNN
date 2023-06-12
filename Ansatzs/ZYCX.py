import pennylane as qml
from Ansatzs.PQCBase import PQCBase

class ZYCXAnsatz(PQCBase):
    def __init__(self, dev, num_bits, num_layers):
        super().__init__(dev)
        self.num_bits = num_bits
        self.num_layers = num_layers

    def layer(self, parameters, pattern):
        """layer-build helper function for ansatz construction"""
        qml.broadcast(qml.RY, wires=range(self.num_bits),
                      pattern="single", parameters=parameters)
        qml.broadcast(qml.CNOT, wires=range(self.num_bits), pattern=pattern)

    def ansatz(self, parameters):
        """PQC w/o input encoding"""
        # slice parameters into num_bits sub-lists
        para_arr = [parameters[i:i + self.num_bits] for i in range(0, len(parameters), self.num_bits)]
        #? reverse the order of the sub-lists, but why?
        para_arr.reverse()

        qml.broadcast(qml.RZ, wires=range(self.num_bits),
                      pattern="single", parameters=para_arr.pop())
        for i in range(self.num_layers):
            pattern = "double" if i % 2 == 0 else "double_odd"
            self.layer(para_arr.pop(), pattern)
        qml.broadcast(qml.RX, wires=range(self.num_bits),
                      pattern="single", parameters=para_arr.pop())

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
            qml.broadcast(qml.RY, wires=range(self.num_bits),
                          pattern="single", parameters=x)
            self.ansatz(parameters)
            return qml.expval(qml.PauliZ(self.num_bits - 1))

        return circuit_real(x, parameters)

