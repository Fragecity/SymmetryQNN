from Symmetries.Symmetry import Symmetry
from Entanglement.WernerUtils import tensor
from Symmetries.WernerSysSwapSymmetry import wernerSymGroup

class WernerSysSwapSymmetry_p1(Symmetry):
    """Here num_qubits means: num_qubits in each subsystem"""
    def __init__(self, num_qubit, global_ob) -> None:
        super().__init__()
        self.group = wernerSymGroup(num_qubit)
        self.observable = global_ob

class WernerSysSwapSymmetry_p2(Symmetry):
    """Here num_qubits means: num_qubits in each subsystem"""
    def __init__(self, num_qubit, global_ob) -> None:
        super().__init__()
        tmp_group = wernerSymGroup(num_qubit)
        #?
        self.group = [tensor(ele, ele) for ele in tmp_group]
        self.observable = global_ob
