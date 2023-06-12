import json
from Entanglement.MultiBits.DsSymGenLoader import genSaveWernerDsSym

save_path = './data/test_ds.pkl'

with open('Config.json') as json_file: # Config_comp.json
        CONFIG = json.load(json_file)

genSaveWernerDsSym(CONFIG, save_path, CUT_COEFF = 0.8)