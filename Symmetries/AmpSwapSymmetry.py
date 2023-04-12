import pennylane as qml
from .Symmetry import Symmetry


class AmpSwapSymmetry(Symmetry):
    """
    # AmpSwapSymmetry that implements the Symmetry interface
    # Defined the deployed symmetry group and the global observable by QCs
    """

    def __init__(self, num_bits) -> None:
        super().__init__()
        self.dev = qml.device('default.qubit', wires=num_bits)
        self.num_bits = num_bits

        @qml.qnode(device=self.dev, interface="torch")
        def swap_circ(i, j):
            qml.SWAP(wires=[i, j])
            return qml.expval(qml.Identity(wires=range(num_bits)))

        #* symmetry group
        swap_ele = qml.matrix(swap_circ)(0, 1)
        self.group = swap_ele

        #* global observable by a QC (Z msmt here)
        @qml.qnode(device=self.dev, diff_method="backprop", interface="torch")
        def observable_circ():
            qml.PauliZ(num_bits - 1)
            return qml.expval(qml.Identity(wires=range(num_bits)))

        self.observable = qml.matrix(observable_circ)()
