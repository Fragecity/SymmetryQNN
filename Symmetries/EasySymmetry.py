from Symmetries.Symmetry import Symmetry

class EasySymmetry(Symmetry):
    """Shortcut definition for convenience\n
    Args:
    @ sym_group: freeGroup of symmetry
    @ global_ob: global observable

    could fill any position as required
    """
    def __init__(self, sym_group=[], global_ob=1) -> None:
        super().__init__()
        self.group = sym_group
        self.observable = global_ob
