import pickle
from Symmetries.WernerSysSwapSymmetry import WernerSysSwapSymmetry
from Entanglement.WernerState.WernerUtils import genWernerDataWithLabel_PL

ds_savePath = './data/test_ds.pkl'

#* data
NUM_DATA = 1
NUM_QUBIT = 1
train_set = genWernerDataWithLabel_PL(NUM_QUBIT, NUM_DATA)
test_set = genWernerDataWithLabel_PL(NUM_QUBIT, 4*NUM_DATA)

#* symmetry
symmetry = WernerSysSwapSymmetry(NUM_QUBIT)

#* save
with open(ds_savePath, 'wb') as f:
    pickle.dump([NUM_QUBIT, train_set, test_set, symmetry], f)