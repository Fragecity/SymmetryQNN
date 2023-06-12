import numpy as np
import json, pickle
from Utils import pauli_group
from Symmetries.EasySymmetry import EasySymmetry
from Entanglement.WernerUtils import witness_PL, genWernerDataWithLabel_PL

ds_savePath = './data/test_ds.pkl'

with open('Config.json') as json_file:
    CONFIG = json.load(json_file)

NUM_PART, NUM_DATA = CONFIG['num_part'], CONFIG['num_data']

#* data
data = genWernerDataWithLabel_PL(NUM_PART, NUM_DATA)
CUT = int(NUM_DATA * 0.8 )
sample_set = data[:CUT]
test_set = data[CUT:]

#* symmetry
sym_group = [np.kron(element, element) for element in pauli_group(NUM_PART)]
W = witness_PL(NUM_PART)
global_ob = np.diag(np.linalg.eigvals(W))
symmetry = EasySymmetry(sym_group, global_ob)

#* save
with open(ds_savePath, 'wb') as f:
    pickle.dump([sample_set, test_set, symmetry], f)