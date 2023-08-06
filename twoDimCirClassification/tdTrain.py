import copy
import datetime
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

from Ansatzs.EasyPQC import EasyPQC
from Symmetries.EasySymmetry import EasySymmetry
from losses import cost, costG, accuracy
from Utils import SWAP, Id, X, Z, tensorOfListOps2
from twoDimCirClassification.QNNLib import dev, enc_func, cir1, cir2

# ! ---------------------------------- Control Panel ----------------------------------
#TODO: seems could add num_params for each circuit

BATCH_SIZE = 32
useShuffleSeed = False  # only for uniform samples (not include uniform2)
SHUFFLE_SEED = 42  # 42

enc_func = enc_func
ansatz = cir1
# ini params
NUM_PARAMS = 8
params_taped = qnp.random.uniform(0, 2 * qnp.pi, size=(NUM_PARAMS), requires_grad=True)
# train
NUM_EPOCH = 1000
LAMBDA = 1
STEP_SIZE = 0.01
opt = qml.AdamOptimizer(stepsize=STEP_SIZE)  # GradientDescentOptimizer
# checking
CHECK_INTERVAL = 1  # interval of epochs for printing training info
plotFlag = "best"  # "best" "final" "off"

recordedPrints = []

def savedPrint(*args, **kwargs):
    recordedPrints.append(' '.join(map(str, args)))
    print(*args, **kwargs)

sgOn = True
saveDate = datetime.datetime.now().strftime("%m%d%H%M")
# ! ---------------------------------- Data ----------------------------------
train_ds = np.load("./Data/trainSet.npy", allow_pickle=True)
test_ds = np.load("./Data/testSet.npy", allow_pickle=True)

NUM_BATCH = len(train_ds) // BATCH_SIZE

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
for i in range(NUM_EPOCH):
    # * batched train
    np.random.shuffle(train_ds)
    if not sgOn:
        bestTrainAccG, bestTrainEpochG, accG, cstG, bestTrainParamsG = 0,0,0,0,np.zeros(NUM_PARAMS)
    for j in range(0, NUM_BATCH):
        batch = train_ds[j * BATCH_SIZE:j * BATCH_SIZE + BATCH_SIZE]
        params_taped = opt.step(lambda v: cost(qnn, v, batch), params_taped)
        if sgOn:
            paramsG_taped = opt.step(lambda v: costG(qnn, v, symmetry, LAMBDA, batch), paramsG_taped)
    # * metric values
    cst = cost(qnn, params_taped, train_ds)
    acc = accuracy(qnn, params_taped, train_ds)
    if sgOn:
        cstG = costG(qnn, paramsG_taped, symmetry, LAMBDA, train_ds)
        accG = accuracy(qnn, paramsG_taped, train_ds)
    # * record the best params
    if i == 1:
        bestTrainAcc, bestTrainAccG = acc, accG
        bestTrainEpoch, bestTrainEpochG = i, i
        bestTrainParams, bestTrainParamsG = params_taped, paramsG_taped
    else:
        if acc > bestTrainAcc:
            bestTrainAcc = acc
            bestTrainEpoch = i
            bestTrainParams = params_taped
        if sgOn:
            if accG > bestTrainAccG:
                bestTrainAccG = accG
                bestTrainEpochG = i
                bestTrainParamsG = paramsG_taped
    # * log
    if (i + 1) % CHECK_INTERVAL == 0 or i < 10:
        savedPrint(f"Eopch: {i + 1:02}, train_ds cst: {cst :0.4f}:, acc: {acc: 0.4f}, cstG: {cstG :0.4f}:, accG: {accG: 0.4f}")
savedPrint("---------------- Finished training ----------------")

# * print final result
acc_train, acc_test = accuracy(qnn, params_taped, train_ds), accuracy(qnn, params_taped, test_ds)
accG_train, accG_test = accuracy(qnn, paramsG_taped, train_ds), accuracy(qnn, paramsG_taped, test_ds)
savedPrint(f"FinalTrain:\nacc_train: {acc_train: 0.4f}, acc_test: {acc_test: 0.4f}\naccG_train: {accG_train: 0.4f}, accG_test: {accG_test: 0.4f}")
# * print best result
bestTrainAccTest, bestTrainAccGTest = accuracy(qnn, bestTrainParams, test_ds), accuracy(qnn, bestTrainParamsG, test_ds)
savedPrint(f"bestTrain:\nbestTrainAcc: {bestTrainAcc: 0.4f}, bestTrainAccTest: {bestTrainAccTest: 0.4f}, at Epoch: {bestTrainEpoch+1}\n\
bestTrainAccG: {bestTrainAccG: 0.4f}, bestTrainAccGTest: {bestTrainAccGTest: 0.4f}, at Epoch: {bestTrainEpochG+1}")
print("\nMain Finished.")

# ! ---------------------------------- Save ----------------------------------
#TODO: could use save Json
# * save settings
savePathRoot = f"./RsRecords/td_{saveDate}_{ansatz.__name__}"
# * save params
np.save(savePathRoot + "_params.npy", np.concatenate((params_taped, paramsG_taped, bestTrainParams, bestTrainParamsG)))
# * save settings and logs
#TODO: seems some of these could be write in a simpler way, combined
with open(savePathRoot + "_log.txt", "w") as f:
    f.write(f"saveDate: {saveDate}\n\n")
    f.write(f"useShuffleSeed: {useShuffleSeed}\nShuffleSeed: {SHUFFLE_SEED}\n")
    f.write(f"Ansatz: {ansatz.__name__}\n")
    f.write(f"\nNUM_BATCH: {NUM_BATCH}\n")
    f.write(f"Num_Epoch: {NUM_EPOCH}\n")
    f.write(f"Step_Size: {STEP_SIZE}\n")
    f.write(f"SG_Weight: {LAMBDA}\n")
    f.write(f"\nLogs:\n")
    for line in recordedPrints:
        f.write(line + "\n")
    f.write("\nOther Info:\n")
    f.write(f"Ob1: \n{ob}")
print("Plot and Save Finished.")