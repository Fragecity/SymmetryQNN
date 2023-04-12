import copy
import pennylane.numpy as np


def generate_freegroup(a_set):
    FreeGroup = list()
    gates_queq = copy.deepcopy(a_set)

    while gates_queq:
        gate = gates_queq.pop()
        if not mat_in_set(gate, FreeGroup):
            FreeGroup.append(gate)
            for s in a_set: gates_queq.append(gate @ s)
        # print(len(FreeGroup))

    return FreeGroup


def mat_in_set(A, Group) -> bool:
    for G in Group:
        if equal(A, G): return True
    return False


def equal(A, B) -> bool:
    B = np.abs(A - B) < 1e-6
    return B.all()
