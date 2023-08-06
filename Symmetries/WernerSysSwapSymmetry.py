from Symmetries.Symmetry import Symmetry
from Utils import dagger
from Entanglement.WernerUtils import pauli_group, swap_2sys, bell_unitary, tensor

def wernerSymGroup(num_qubit):
    """Generate the symmetry group of Werner state
    Args:
        num_qubit (int): number of qubits in each subsystem
    """
    pau_group = pauli_group(num_qubit)
    return [tensor(ele, ele) for ele in pau_group] #TODO: though SWAP is also one possible choice, but no need?

class WernerSysSwapSymmetry(Symmetry):
    """Here num_qubits means: num_qubits in each subsystem

    PAT: this has a pre-defined observable, not using what the circuit provides
    """
    def __init__(self, num_qubit) -> None:
        super().__init__()
        self.group = wernerSymGroup(num_qubit)

        SWAP = swap_2sys(num_qubit)
        U = bell_unitary(num_qubit)
        self.observable = U @ SWAP @ dagger(U)
