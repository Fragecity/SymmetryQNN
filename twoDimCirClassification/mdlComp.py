import re
import os
import numpy as np
import pennylane as qml
from functools import partial
from autograd.numpy.numpy_boxes import ArrayBox

from Ansatzs.EasyPQC import EasyPQC
from Symmetries.EasySymmetry import EasySymmetry
from Cleaners import genDecisionArea
from Utils import SWAP, Id, X, Z, tensorOfListOps2
from twoDimCirClassification.QNNLib import dev, enc_func, cir1, cir2, cir2wl

#3 td_08040530_cir1
#5 td_0814-03-39-28_cir2wl_5L_adam0.2_sg1
#10 td_0814-03-40-05_cir2wl_10L_adam0.2_sg1


# recordPath
recordPathRoot = "./RsRecords_pureSG"
recordId = "td_0817-09-37-35_small_cir2wl_5L_adam0.1_sg1_pureSG"

def sgValue(qnn, params, symmetry):
    U = np.squeeze(qml.matrix(qnn.U_circ)(params))
    print("U:")
    print(U)

    if isinstance(U, ArrayBox):
        U = np.array(U._value)

    return symmetry.symmetry_guidance(U)

def genSGValues(recordId):
    fileName = recordId + "_log.txt"
    with open(os.path.join(recordPathRoot, fileName), 'r') as file:
        content = file.read()
        patterns = [
            "Ansatz:.*?\n",
            "NUM_LAYERS:.*?\n"
        ]
        extracted_data = []
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted_data.append(match.group())
    AnsatzsName = extracted_data[0].split(":")[1].strip()
    if len(extracted_data) == 2:
        NUM_LAYERS = int(extracted_data[1].split(":")[1])
    if AnsatzsName == "cir2wl":
        ansatz = partial(cir2wl, num_layers=NUM_LAYERS)
    elif AnsatzsName == "cir1":
        ansatz = cir1
    elif AnsatzsName == "cir2":
        ansatz = cir2

    ob = tensorOfListOps2(Z, Z, Z, Z)
    O = qml.Hermitian(ob, wires=range(4))
    qnn = EasyPQC(dev, enc_func, ansatz, O, O)

    paramsPath = os.path.join(recordPathRoot, recordId + "_params.npy")

    loaded_data = np.load(paramsPath, allow_pickle=True)
    individual_size = len(loaded_data) // 4  # Divide by 4 since there are four arrays
    # Split the loaded data back into individual arrays
    params_taped_loaded = loaded_data[:individual_size]
    paramsG_taped_loaded = loaded_data[individual_size:2 * individual_size]
    bestTrainParams_loaded = loaded_data[2 * individual_size:3 * individual_size]
    bestTrainParamsG_loaded = loaded_data[3 * individual_size:4 * individual_size]

    symGroup = [tensorOfListOps2(Id, Id, Id, Id), np.kron(SWAP, SWAP)]
    symmetry = EasySymmetry(symGroup, ob)

    sg1 = sgValue(qnn, bestTrainParams_loaded, symmetry)
    sg2 = sgValue(qnn, bestTrainParamsG_loaded, symmetry)

    return sg1, sg2

sg1, sg2 = genSGValues(recordId)
print(f"sg1: {sg1}, sg2: {sg2}")

