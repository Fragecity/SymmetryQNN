import json, pickle, time
from Entanglement.MultiBits.Trainer import iterCompare

ds_loadPath = './data/test_ds.pkl'

lcl_time = time.strftime("%m-%d %H-%M", time.localtime())
savePath = './data_comp/result ' + lcl_time + '.pkl'

with open('Config.json') as json_file:
    CONFIG = json.load(json_file)
with open(ds_loadPath, 'rb') as f:
    [sample_set, test_set, symmetry] = pickle.load(f)

layer_list = range(1,6)
iter_list = range(2,15)
res = iterCompare(layer_list, iter_list, CONFIG, sample_set, symmetry)

with open(savePath, 'wb') as f:
    pickle.dump(res, f)