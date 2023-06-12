"""Gen and Save Werner_dataSet and Symmetry"""

import pickle
from Utils import pauli_group
from Utils import tensorOfListOps as tensor
from Symmetries.EasySymmetry import EasySymmetry
from Entanglement.WernerState.WernerUtils import genWernerDataWithLabel as gener_data


def genSaveWernerDsSym(CONFIG, save_path, CUT_COEFF = 0.8):
    """Generate then Save Werner_dataSet and Symmetry as: [sample_set, test_set, symmetry]"""
    #* ------------------- DataSet ------------------- #
    NUM_PART = CONFIG['num_part']
    NUM_DATA = CONFIG['num_data']

    data = gener_data(NUM_PART, NUM_DATA)
    CUT = int(NUM_DATA * CUT_COEFF)
    sample_set = data[:CUT]
    test_set = data[CUT:]

    #* ------------------- Symmetry ------------------- #
    subgroup = pauli_group(NUM_PART)
    sym_group = [tensor(element, element) for element in subgroup]
    symmetry = EasySymmetry(sym_group=sym_group)

    #* ------------------- Save ------------------- #
    with open(save_path, 'wb') as f:
        pickle.dump([sample_set, test_set, symmetry], f)


def loadWernerDsSym(load_path):
    """Load Werner_dataSet and Symmetry as: [sample_set, test_set, symmetry]"""
    with open(load_path, 'rb') as f:
        [sample_set, test_set, symmetry] = pickle.load(f)
    return sample_set, test_set, symmetry