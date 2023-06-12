import pickle, json
from Entanglement.MultiBits.Trainer import train1

ds_loadPath = './data/test_ds.pkl'

pre_para_path = './data/result cost 2023-03-24 16-28 (star).pkl'

with open('Config.json') as json_file:
    CONFIG = json.load(json_file)
with open(ds_loadPath, 'rb') as f:
    sample_set, test_set, symmetry = pickle.load(f)
with open(pre_para_path, 'rb') as f:
    [x_lst, _] = pickle.load(f)


train1(CONFIG, sample_set, symmetry, pre_para=x_lst)