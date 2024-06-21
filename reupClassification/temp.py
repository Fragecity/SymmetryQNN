import os, sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml

# from Cleaners import genDecisionArea
from Ansatzs.EasyPQC import EasyPQC
from Symmetries.EasySymmetry import EasySymmetry
from Utils import SWAP, Id, X, Z, tensorOfListOps2, Permutationij
from reupClassification.losses import cost, costTwiredLoss, accuracy, sgRecord
from reupClassification.Cleaners import (
    optimizerSelector,
    dsLoader,
    genDecisionArea,
    optmizedParam,
    sgwScheduler,
)
from reupClassification.QNNRepo.QNNLib import (
    enc_func,
    cir1,
    cir2,
    cir2wl,
    cir_testSG,
    cir2wl_cz,
    enc_func_2d_even,
)
from reupClassification.QNNRepo.QNNLibs import type2MC, type2MC_multiEven
from TrainUtils.AdamOptmizer import AdamOptimizer


dev = qml.device("default.mixed", wires=6)
ansatz = type2MC_multiEven(5, 6).circuit
# train.circuit
NUM_BATCH = 12  # PAT not make some samples be ignored
# ! ---------------------------------- QNN and Symmetry ----------------------------------

ob1 = tensorOfListOps2(Z, Id, Id, Id, Id, Id)
ob2 = tensorOfListOps2(Id, Id, Id, Z, Id, Id)
symGroup = [
    tensorOfListOps2(Id, Id, Id, Id, Id, Id),
    np.kron(SWAP, np.kron(SWAP, SWAP)),
]
O1 = qml.Hermitian(ob1, wires=range(6))
O2 = qml.Hermitian(ob2, wires=range(6))
qnn_ob1 = EasyPQC(dev, enc_func, ansatz, O1, O1)
qnn_ob2 = EasyPQC(dev, enc_func, ansatz, O2, O2)


fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
# fig.suptitle(recordId)


bestTrainParams = np.load("./reupClassification/RsRecords/td_0605-09-49-14_small_100_cir2wl_multiEven_5L_adam0.1/params.npy")
genDecisionArea(
    qnn_ob1, qnn_ob2, bestTrainParams, num_sample=50, title="loss", ax=axs[0], showPlot=True
)
# genDecisionArea(
#     qnn_ob1, qnn_ob2, bestTrainParamsG, num_sample=50, title="lossG", ax=axs[1]
# )

# plt.tight_layout()
# fig.savefig(savePathRoot + "combinedDecisionArea.png")
# plt.show()

# # * plot training curves
# fig, axs = plt.subplots(
#     1, 3, figsize=(16, 5)
# )  # figsize can be adjusted as per your preference
# fig.suptitle(recordId + f"num_batch: {NUM_BATCH}")

# # * cost curves
# axs[0].plot(range(1, cnt + 1), cst_List, label="cost")
# axs[0].set_xlabel(f"batch_cnt, bath_size: {BATCH_SIZE}")
# axs[0].set_ylabel("cost")
# axs[0].set_title(f"loss curve")
# axs[0].legend()

# axs[1].plot(range(1, cnt + 1), cstG_List, label="costG")
# axs[1].set_xlabel("batch_cnt")
# axs[1].set_ylabel("costG")
# axs[1].set_title(f"lossG curve")
# axs[1].legend()

# # accuracy curves
# axs[2].plot(range(1, cnt + 1), acc_List, label="acc")
# axs[2].plot(range(1, cnt + 1), accG_List, label="accG")
# axs[2].set_xlabel("batch_cnt")
# axs[2].set_ylabel("acc")
# axs[2].set_title(f"acc curves")
# axs[2].legend()