import os, sys
import copy
import datetime
import numpy as np
import pennylane as qml
from functools import partial
from pennylane import numpy as qnp
import matplotlib.pyplot as plt


sys.path.append("./")
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

# ! ---------------------------------- Control Panel ----------------------------------
# TODO: seems could add num_params for each circuit
savePath = "./reupClassification/RsRecords/"
NUM_EPOCH = 1
# * model
NUM_LAYERS = 6
NUM_QUBITS = 6  # 4
dev = qml.device("default.mixed", wires=NUM_QUBITS)
# ansatz = partial(cir2wl, num_layers=NUM_LAYERS) # cir_testSG, partial(cir2wl_cz, num_layers=NUM_LAYERS)
midcir = type2MC_multiEven(NUM_LAYERS, NUM_QUBITS)
# train
OPTIMIZER = "adam"  # 'adam' 'gd'
STEP_SIZE = 0.1
STEP_SIZE_SG = 0.1
# data
DS_TYPE = "test"  # 'basic' 'small_100' 'test'
BATCH_SIZE = 16  # 32(small) 6(Small30) 3(verySmall) small60/small60_xhalf(12)
# * model
ansatz = midcir.circuit
# NUM_PARAMS = 80     # num of ini params
enc_func = partial(enc_func_2d_even, num_qubits=NUM_QUBITS)
params_taped = qnp.random.uniform(
    0, 2 * qnp.pi, size=midcir.num_params, requires_grad=True
)  # qnp.array([4.07795418, 2.01460455], requires_grad=True)
# * data
useShuffleSeed = False  # only for uniform samples (not include uniform2)
SHUFFLE_SEED = 42  # 42
# * train
opt = AdamOptimizer(lr=STEP_SIZE)
optG = AdamOptimizer(lr=STEP_SIZE_SG)
# opt = optimizerSelector(OPTIMIZER, STEP_SIZE)
# optG = optimizerSelector(OPTIMIZER, STEP_SIZE)
# checking
CHECK_INTERVAL = 1  # interval of epochs for printing training info
plotFlag = "best"  # "best" "final" "off"
recordedPrints = []


def savedPrint(*args, **kwargs):
    recordedPrints.append(" ".join(map(str, args)))
    print(*args, **kwargs)


sgOn = True
saveDate = datetime.datetime.now().strftime("%m%d-%H-%M-%S")
recordId = f"td_{saveDate}_{DS_TYPE}_{midcir.name}_{NUM_LAYERS}L_{OPTIMIZER}{STEP_SIZE}"
savePathRoot = savePath + recordId + "/"
print(f"Start {savePathRoot}")
# savePathRoot = f"./RsRecords/testSG/td_{saveDate}_{DS_TYPE}_{ansatz.__name__}_{NUM_LAYERS}L_{OPTIMIZER}{STEP_SIZE}_sg{LAMBDA}_pureSG"
# ! ---------------------------------- Data ----------------------------------
train_ds, test_ds = dsLoader(DS_TYPE)

NUM_BATCH = len(train_ds) // BATCH_SIZE  # PAT not make some samples be ignored

# ! ---------------------------------- QNN and Symmetry ----------------------------------
if NUM_QUBITS == 4:
    ob1 = tensorOfListOps2(Z, Id, Id, Id)
    ob2 = tensorOfListOps2(Id, Z, Id, Id)
    symGroup = [tensorOfListOps2(Id, Id, Id, Id), np.kron(SWAP, SWAP)]
elif NUM_QUBITS == 6:
    ob1 = tensorOfListOps2(Z, Id, Id, Id, Id, Id)
    ob2 = tensorOfListOps2(Id, Id, Id, Id, Id, Z)
    symGroup = [
        tensorOfListOps2(Id, Id, Id, Id, Id, Id),
        np.kron(SWAP, np.kron(SWAP, SWAP)),
    ]
O1 = qml.Hermitian(ob1, wires=range(NUM_QUBITS))
O2 = qml.Hermitian(ob2, wires=range(NUM_QUBITS))

qnn_ob1 = EasyPQC(dev, enc_func, ansatz, O1, O1)
qnn_ob2 = EasyPQC(dev, enc_func, ansatz, O2, O2)

symmetry_ob1 = EasySymmetry(symGroup, ob1)
symmetry_ob2 = EasySymmetry(symGroup, ob2)

# ! ---------------------------------- Train and log ----------------------------------
# * ini of params and record
paramsG_taped = copy.deepcopy(params_taped)
bestTrainAcc, bestTrainAccG, bestTrainEpoch, bestTrainEpochG = 0, 0, 0, 0
bestTrainParams, bestTrainParamsG = params_taped, paramsG_taped

savedPrint("Before training")
savedPrint("Accuracy on train_ds: ", accuracy(qnn_ob1, qnn_ob2, params_taped, train_ds))
savedPrint("Accuracy on test_ds: ", accuracy(qnn_ob1, qnn_ob2, params_taped, test_ds))

savedPrint("---------------- Start training ----------------")
cnt = 0
cst_List, cstG_List, acc_List, accG_List = [], [], [], []
for i in range(NUM_EPOCH):
    # * batched train
    np.random.shuffle(train_ds)
    if not sgOn:
        bestTrainAccG, bestTrainEpochG, accG, cstG, bestTrainParamsG = (
            0,
            0,
            0,
            0,
            np.zeros(midcir.num_params),
        )
    for j in range(0, NUM_BATCH):
        # time0 = datetime.datetime.now()
        cnt += 1
        batch = train_ds[j * BATCH_SIZE : j * BATCH_SIZE + BATCH_SIZE]
        # params_taped = opt.step(lambda v: cost(v, qnn, batch), params_taped)
        params_taped = opt.step(
            partial(cost, qnn_ob1=qnn_ob1, qnn_ob2=qnn_ob2, data_batch=batch),
            params_taped,
        )
        cst = cost(params_taped, qnn_ob1, qnn_ob2, train_ds)

        acc = accuracy(
            qnn_ob1, qnn_ob2, params_taped, train_ds
        )  # comment this line for faster training

        cst_List.append(cst)
        acc_List.append(acc)  # comment this line for faster training
        if sgOn:
            paramsG_taped = optG.step(
                partial(
                    costTwiredLoss,
                    qnn_ob1=qnn_ob1,
                    qnn_ob2=qnn_ob2,
                    symmetry_ob1=symmetry_ob1,
                    symmetry_ob2=symmetry_ob2,
                    data_batch=batch,
                ),
                paramsG_taped,
            )
            # print(params_taped)
            cstG = costTwiredLoss(
                paramsG_taped, qnn_ob1, qnn_ob2, symmetry_ob1, symmetry_ob2, train_ds
            )

            accG = accuracy(
                qnn_ob1, qnn_ob2, paramsG_taped, train_ds
            )  # comment this line for faster training

            cstG_List.append(cstG)
            accG_List.append(accG)  # comment this line for faster training
        # print(paramsG_taped[0:2])
        # * record the best params
        if cnt == 1:
            bestTrainAcc, bestTrainAccG = acc, accG
            bestTrainEpoch, bestTrainEpochG, bestTrainBatch, bestTrainBatchG = (
                i,
                i,
                cnt,
                cnt,
            )
            bestTrainParams, bestTrainParamsG = params_taped, paramsG_taped
        else:
            if acc >= bestTrainAcc:
                bestTrainAcc = acc
                bestTrainEpoch = i
                bestTrainParams = params_taped
                bestTrainBatch = cnt
            if sgOn:
                if accG >= bestTrainAccG:
                    bestTrainAccG = accG
                    bestTrainEpochG = i
                    bestTrainParamsG = paramsG_taped
                    bestTrainBatchG = cnt
        # * log
        # if (i + 1) % CHECK_INTERVAL == 0 or i < 10:
        if cnt % CHECK_INTERVAL == 0 or i < 10:
            # print(f"time_cost: {datetime.datetime.now() - time0}")
            savedPrint(
                f"Epoch: {i + 1:02}, cnt: {cnt}, train_ds cst: {cst :0.4f}:, acc: {acc: 0.4f}, cstG: {cstG :0.4f}:, accG: {accG: 0.4f}"
            )
savedPrint("---------------- Finished training ----------------")

# * print final result
acc_train, acc_test = accuracy(qnn_ob1, qnn_ob2, params_taped, train_ds), accuracy(
    qnn_ob1, qnn_ob2, params_taped, test_ds
)
accG_train, accG_test = accuracy(qnn_ob1, qnn_ob2, paramsG_taped, train_ds), accuracy(
    qnn_ob1, qnn_ob2, paramsG_taped, test_ds
)
savedPrint(
    f"FinalTrain:\nacc_train: {acc_train: 0.4f}, acc_test: {acc_test: 0.4f}\naccG_train: {accG_train: 0.4f}, accG_test: {accG_test: 0.4f}"
)
# * print best result
bestTrainAccTest, bestTrainAccGTest = accuracy(
    qnn_ob1, qnn_ob2, bestTrainParams, test_ds
), accuracy(qnn_ob1, qnn_ob2, bestTrainParamsG, test_ds)
savedPrint(
    f"bestTrain:\nbestTrainAcc: {bestTrainAcc: 0.4f}, bestTrainAccTest: {bestTrainAccTest: 0.4f}, at Epoch: {bestTrainEpoch+1}, at Batch: {bestTrainBatch}\n\
bestTrainAccG: {bestTrainAccG: 0.4f}, bestTrainAccGTest: {bestTrainAccGTest: 0.4f}, at Epoch: {bestTrainEpochG+1} at Batch: {bestTrainBatchG}"
)
print("\nMain Finished.")

# ! ---------------------------------- Save ----------------------------------
# TODO: could use save Json
# * save settings
dir_name = os.path.dirname(savePathRoot)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
# * save params
np.save(
    savePathRoot + "params.npy",
    np.concatenate((params_taped, paramsG_taped, bestTrainParams, bestTrainParamsG)),
)
# * save settings and logs
# TODO: seems some of these could be write in a simpler way, combined
log_contents_pre = f"""saveDate: {saveDate}

Ansatz: {midcir.name}
NUM_LAYERS: {NUM_LAYERS}
Num_Epoch: {NUM_EPOCH}

Optimizer: {OPTIMIZER}
Step_Size: {STEP_SIZE}
NUM_BATCH: {NUM_BATCH}

DS_TYPE: {DS_TYPE}
useShuffleSeed: {useShuffleSeed}
ShuffleSeed: {SHUFFLE_SEED}

Logs:
"""
with open(savePathRoot + "log.txt", "w") as f:
    f.write(log_contents_pre)
    for line in recordedPrints:
        f.write(line + "\n")
    f.write("\nOther Info:\n")
    f.write(f"Ob1: \n{ob1}")
    f.write(f"Ob2: \n{ob2}")
print("Plot and Save Finished.")
print("saveDate: ", saveDate)

# * for easy checking
log_contents_check = f"""DS_TYPE: {DS_TYPE}
Ansatz: {midcir.name}
NUM_LAYERS: {NUM_LAYERS}
Optimizer: {OPTIMIZER}
Step_Size: {STEP_SIZE}"""
print(log_contents_check)
print(savePathRoot)

# ! ---------------------------------- Plots ----------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
fig.suptitle(recordId)

genDecisionArea(
    qnn_ob1, qnn_ob2, bestTrainParams, num_sample=50, title="loss", ax=axs[0]
)
genDecisionArea(
    qnn_ob1, qnn_ob2, bestTrainParamsG, num_sample=50, title="lossG", ax=axs[1]
)

plt.tight_layout()
fig.savefig(savePathRoot + "combinedDecisionArea.png")
# plt.show()

# * plot training curves
fig, axs = plt.subplots(
    1, 3, figsize=(16, 5)
)  # figsize can be adjusted as per your preference
fig.suptitle(recordId + f"num_batch: {NUM_BATCH}")

# * cost curves
axs[0].plot(range(1, cnt + 1), cst_List, label="cost")
axs[0].set_xlabel(f"batch_cnt, bath_size: {BATCH_SIZE}")
axs[0].set_ylabel("cost")
axs[0].set_title(f"loss curve")
axs[0].legend()

axs[1].plot(range(1, cnt + 1), cstG_List, label="costG")
axs[1].set_xlabel("batch_cnt")
axs[1].set_ylabel("costG")
axs[1].set_title(f"lossG curve")
axs[1].legend()

# accuracy curves
axs[2].plot(range(1, cnt + 1), acc_List, label="acc")
axs[2].plot(range(1, cnt + 1), accG_List, label="accG")
axs[2].set_xlabel("batch_cnt")
axs[2].set_ylabel("acc")
axs[2].set_title(f"acc curves")
axs[2].legend()

# Save the combined figure
fig.savefig(savePathRoot + "learningCurves.png")
# plt.show()

# * save data of curves
np.save(savePathRoot + "cst.npy", np.array(cst_List))
np.save(savePathRoot + "cstG.npy", np.array(cstG_List))
np.save(savePathRoot + "acc.npy", np.array(acc_List))
np.save(savePathRoot + "accG.npy", np.array(accG_List))

print("training curves: plot and save finished")
