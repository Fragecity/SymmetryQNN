import pickle, json
from Entanglement.WernerState.MultiBits.Trainer import train1

ds_loadPath = './data/test_ds.pkl'

with open('Config.json') as json_file:
    CONFIG = json.load(json_file)
with open(ds_loadPath, 'rb') as f:
    sample_set, test_set, symmetry = pickle.load(f)

train1(CONFIG, sample_set, symmetry)