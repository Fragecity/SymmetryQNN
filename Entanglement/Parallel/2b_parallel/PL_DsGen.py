import pickle
from Symmetries.EasySymmetry import EasySymmetry
from Utils import pauli_group, dagger
from Utils import tensorOfListOps as tensor
from Entanglement.WernerUtils import swap_2sys, bell_unitary, genWernerDataWithLabel_PL

ds_savePath = './data/test_ds.pkl'

NUM_DATA = 2
NUM_QUBIT = 1
train_set = genWernerDataWithLabel_PL(NUM_QUBIT, NUM_DATA)
test_set = genWernerDataWithLabel_PL(NUM_QUBIT, 16*NUM_DATA)

pau_group = pauli_group(NUM_QUBIT)
sym_group = [tensor(ele, ele, ele, ele, ele, ele) for ele in pau_group]

SWAP = swap_2sys(NUM_QUBIT)
SWAP = tensor(SWAP, SWAP, SWAP)
U = bell_unitary(NUM_QUBIT)
U = tensor(U,U,U)
global_ob = U@SWAP@dagger(U)

symmetry = EasySymmetry(sym_group, global_ob)

with open(ds_savePath, 'wb') as f:
    pickle.dump([NUM_QUBIT, train_set, test_set, symmetry], f)