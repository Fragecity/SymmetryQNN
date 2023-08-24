import os
import copy
import datetime
import numpy as np
import pennylane as qml
from functools import partial
from pennylane import numpy as qnp
import matplotlib.pyplot as plt

from Ansatzs.EasyPQC import EasyPQC
from Symmetries.EasySymmetry import EasySymmetry
from losses import cost, costG, accuracy, sgRecord
from Cleaners import optimizerSelector, dsLoader, genDecisionArea, optmizedParam, sgwScheduler
from Utils import SWAP, Id, X, Z, tensorOfListOps2
from QNNLib import dev, enc_func, cir1, cir2, cir2wl, cir_testSG, cir2wl_cz
from QNNRepo.QNNLibs import type2MC
from TrainUtils.AdamOptmizer import AdamOptimizer

# ! ---------------------------------- Control Panel ----------------------------------
#TODO: seems could add num_params for each circuit
savePath = "./RsRecords/" + "a_0824/findingAdvan" + "/"
NUM_EPOCH = 50
#* model
NUM_LAYERS = 5
# ansatz = partial(cir2wl, num_layers=NUM_LAYERS) # cir_testSG, partial(cir2wl_cz, num_layers=NUM_LAYERS)
midcir = type2MC(NUM_LAYERS)
# train
OPTIMIZER = 'adam' # 'adam' 'gd'
STEP_SIZE = 0.01
TYPE_SGW = 'linear1' # 'constant' 'linear1' 'postLinear2d3' 'postLinear3d4'
LAMBDA = 8 # SG weight
# data
DS_TYPE = 'small_tl' # '1500' 'small' 'small_tl' 'Small_tlhalf' 'verySmall' 'Small30' 'small60 'small60_xhalf'
BATCH_SIZE = 32 #32(small) 6(Small30) 3(verySmall) small60/small60_xhalf(12)
#* model
ansatz = midcir.circuit
# NUM_PARAMS = 80     # num of ini params
enc_func = enc_func
params_taped = qnp.random.uniform(0, 2 * qnp.pi, size=(midcir.num_params), requires_grad=True) # qnp.array([4.07795418, 2.01460455], requires_grad=True)
#* data
useShuffleSeed = False  # only for uniform samples (not include uniform2)
SHUFFLE_SEED = 42  # 42
#* train
opt = AdamOptimizer(lr=STEP_SIZE)
optG = AdamOptimizer(lr=STEP_SIZE)
# opt = optimizerSelector(OPTIMIZER, STEP_SIZE)
# optG = optimizerSelector(OPTIMIZER, STEP_SIZE)
# checking
CHECK_INTERVAL = 1  # interval of epochs for printing training info
plotFlag = "best"  # "best" "final" "off"
recordedPrints = []

def savedPrint(*args, **kwargs):
    recordedPrints.append(' '.join(map(str, args)))
    print(*args, **kwargs)

sgOn = True
saveDate = datetime.datetime.now().strftime("%m%d-%H-%M-%S")
recordId = f"td_{saveDate}_{DS_TYPE}_{midcir.name}_{NUM_LAYERS}L_{OPTIMIZER}{STEP_SIZE}_sg{LAMBDA}{TYPE_SGW}"
savePathRoot = savePath + recordId + "/"
print(f"Start {savePathRoot}")
# savePathRoot = f"./RsRecords/testSG/td_{saveDate}_{DS_TYPE}_{ansatz.__name__}_{NUM_LAYERS}L_{OPTIMIZER}{STEP_SIZE}_sg{LAMBDA}_pureSG"
# ! ---------------------------------- Data ----------------------------------
train_ds, test_ds = dsLoader(DS_TYPE)

NUM_BATCH = len(train_ds) // BATCH_SIZE # PAT not make some samples be ignored

# ! ---------------------------------- QNN and Symmetry ----------------------------------

ob = tensorOfListOps2(Z, Z, Z, Z)
O = qml.Hermitian(ob, wires=range(4))

qnn = EasyPQC(dev, enc_func, ansatz, O, O)
symGroup = [tensorOfListOps2(Id,Id,Id,Id), np.kron(SWAP, SWAP)]
symmetry = EasySymmetry(symGroup, ob)

# ! ---------------------------------- Train and log ----------------------------------
# * ini of params and record
paramsG_taped = copy.deepcopy(params_taped)
bestTrainAcc, bestTrainAccG, bestTrainEpoch, bestTrainEpochG = 0, 0, 0, 0
bestTrainParams, bestTrainParamsG = params_taped, paramsG_taped

savedPrint("Before training")
savedPrint("Accuracy on train_ds: ", accuracy(qnn, params_taped, train_ds))
savedPrint("Accuracy on test_ds: ", accuracy(qnn, params_taped, test_ds))

savedPrint("---------------- Start training ----------------")
cnt = 0
cst_List, cstG_List, acc_List, accG_List = [], [], [], []
for i in range(NUM_EPOCH):
    # * batched train
    np.random.shuffle(train_ds)
    if not sgOn:
        bestTrainAccG, bestTrainEpochG, accG, cstG, bestTrainParamsG = 0,0,0,0,np.zeros(midcir.num_params)
    for j in range(0, NUM_BATCH):
        # time0 = datetime.datetime.now()
        cnt += 1
        batch = train_ds[j * BATCH_SIZE:j * BATCH_SIZE + BATCH_SIZE]
        # params_taped = opt.step(lambda v: cost(v, qnn, batch), params_taped)
        params_taped = opt.step(partial(cost, qnn=qnn, data_batch=batch), params_taped)
        cst = cost(params_taped, qnn, train_ds)
        acc = accuracy(qnn, params_taped, train_ds)
        cst_List.append(cst)
        acc_List.append(acc)
        if sgOn:
            # paramsG_taped = optG.step(lambda v: costG(v, qnn, symmetry, LAMBDA, batch), paramsG_taped) # sgwScheduler(i+1) (i+1)/NUM_EPOCH*LAMBDA
            paramsG_taped = optG.step(partial(costG, qnn=qnn, symmetry=symmetry, lamd=sgwScheduler(TYPE_SGW, i, NUM_EPOCH, LAMBDA),
                                              data_batch=batch), paramsG_taped)
            cstG = costG(paramsG_taped, qnn, symmetry, sgwScheduler(TYPE_SGW, i+1, NUM_EPOCH, LAMBDA), train_ds, checkMode=True)
            accG = accuracy(qnn, paramsG_taped, train_ds)
            cstG_List.append(cstG)
            accG_List.append(accG)
    # print(paramsG_taped[0:2])
    # * record the best params
        if cnt == 1:
            bestTrainAcc, bestTrainAccG = acc, accG
            bestTrainEpoch, bestTrainEpochG, bestTrainBatch, bestTrainBatchG = i, i, cnt, cnt
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
            savedPrint(f"Eopch: {i + 1:02}, cnt: {cnt}, train_ds cst: {cst :0.4f}:, acc: {acc: 0.4f}, cstG: {cstG :0.4f}:, accG: {accG: 0.4f}")
savedPrint("---------------- Finished training ----------------")

# * print final result
acc_train, acc_test = accuracy(qnn, params_taped, train_ds), accuracy(qnn, params_taped, test_ds)
accG_train, accG_test = accuracy(qnn, paramsG_taped, train_ds), accuracy(qnn, paramsG_taped, test_ds)
savedPrint(f"FinalTrain:\nacc_train: {acc_train: 0.4f}, acc_test: {acc_test: 0.4f}\naccG_train: {accG_train: 0.4f}, accG_test: {accG_test: 0.4f}")
# * print best result
bestTrainAccTest, bestTrainAccGTest = accuracy(qnn, bestTrainParams, test_ds), accuracy(qnn, bestTrainParamsG, test_ds)
savedPrint(f"bestTrain:\nbestTrainAcc: {bestTrainAcc: 0.4f}, bestTrainAccTest: {bestTrainAccTest: 0.4f}, at Epoch: {bestTrainEpoch+1}, at Batch: {bestTrainBatch}\n\
bestTrainAccG: {bestTrainAccG: 0.4f}, bestTrainAccGTest: {bestTrainAccGTest: 0.4f}, at Epoch: {bestTrainEpochG+1} at Batch: {bestTrainBatchG}")
print("\nMain Finished.")

# ! ---------------------------------- Save ----------------------------------
#TODO: could use save Json
# * save settings
dir_name = os.path.dirname(savePathRoot)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
# * save params
np.save(savePathRoot + "params.npy", np.concatenate((params_taped, paramsG_taped, bestTrainParams, bestTrainParamsG)))
# * save settings and logs
#TODO: seems some of these could be write in a simpler way, combined
log_contents_pre = f"""saveDate: {saveDate}

Ansatz: {midcir.name}
NUM_LAYERS: {NUM_LAYERS}
Num_Epoch: {NUM_EPOCH}
TYPE_SGW: {TYPE_SGW}
SG_Weight: {LAMBDA}

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
    f.write(f"Ob1: \n{ob}")
print("Plot and Save Finished.")
print("saveDate: ", saveDate)

#* for easy checking
log_contents_check = f"""DS_TYPE: {DS_TYPE}
Ansatz: {midcir.name}
NUM_LAYERS: {NUM_LAYERS}
TYPE_SGW: {TYPE_SGW}
SG_Weight: {LAMBDA}
Optimizer: {OPTIMIZER}
Step_Size: {STEP_SIZE}"""
print(log_contents_check)
print(savePathRoot)

# ! ---------------------------------- Plots ----------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
fig.suptitle(recordId)

genDecisionArea(qnn, bestTrainParams, num_sample=50, ax=axs[0])
axs[0].set_title(f"loss")
genDecisionArea(qnn, bestTrainParamsG, num_sample=50, ax=axs[1])
axs[1].set_title(f"lossG")

plt.tight_layout()
fig.savefig(savePathRoot + "combinedDecisionArea.png")
plt.show()

# * plot training curves
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # figsize can be adjusted as per your preference
fig.suptitle(recordId+f"num_batch: {NUM_BATCH}")

#* cost curves
axs[0,0].plot(range(1, cnt+1), cst_List, label="cost")
axs[0,0].set_xlabel(f"batch_cnt, bath_size: {BATCH_SIZE}")
axs[0,0].set_ylabel("cost")
axs[0,0].set_title(f"loss curve")
axs[0,0].legend()

axs[0,1].plot(range(1, cnt+1), cstG_List, label="costG")
axs[0,1].set_xlabel("batch_cnt")
axs[0,1].set_ylabel("costG")
axs[0,1].set_title(f"lossG curve")
axs[0,1].legend()

# sg curve
axs[1,0].plot(range(1, sgRecord['cnt']+1), sgRecord['valList'], label="sgVal")
axs[1,0].set_xlabel("batch_cnt")
axs[1,0].set_ylabel("sgVal")
axs[1,0].set_title(f"sg curve")
axs[1,0].legend()

# accuracy curves
axs[1,1].plot(range(1, cnt+1), acc_List, label="acc")
axs[1,1].plot(range(1, cnt+1), accG_List, label="accG")
axs[1,1].set_xlabel("batch_cnt")
axs[1,1].set_ylabel("acc")
axs[1,1].set_title(f"acc curves")
axs[1,1].legend()

# Save the combined figure
fig.savefig(savePathRoot + "learningCurves.png")
plt.show()

#* save data of curves
np.save(savePathRoot + "cst.npy", np.array(cst_List))
np.save(savePathRoot + "cstG.npy", np.array(cstG_List))
np.save(savePathRoot + "acc.npy", np.array(acc_List))
np.save(savePathRoot + "accG.npy", np.array(accG_List))

print("training curves: plot and save finished")
