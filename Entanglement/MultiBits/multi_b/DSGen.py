import json
from Entanglement.WernerState.MultiBits.DsSymGenLoader import genSaveWernerDsSym

save_path = './data/test_ds.pkl'

with open('Config.json') as json_file:
        CONFIG = json.load(json_file)

genSaveWernerDsSym(CONFIG, save_path, CUT_COEFF = 0.8)