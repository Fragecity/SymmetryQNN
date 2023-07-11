from Symmetries.Symmetry import Symmetry
from Utils import dagger
from Entanglement.WernerUtils import pauli_group, swap_2sys, bell_unitary, tensor


class WernerSysSwapSymmetry(Symmetry):
    """Here num_qubits means: num_qubits in each subsystem"""
    def __init__(self, num_qubit) -> None:
        super().__init__()
        pau_group = pauli_group(num_qubit)
        self.group = [tensor(ele, ele) for ele in pau_group] #TODO: though SWAP is also one possible choice, but no need?

        SWAP = swap_2sys(num_qubit)
        U = bell_unitary(num_qubit)
        self.observable = U @ SWAP @ dagger(U)
