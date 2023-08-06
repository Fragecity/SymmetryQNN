import copy
import datetime
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

from ESSUtils import sigmoid
from Ansatzs.EasyPQC import EasyPQC
from losses import cost, costG, accuracy
from Utils import SWAP, SWAPP, Id, X, Z, tensorOfListOps2
from Entanglement.ExpandedStatesSet.ESSUtils import dm_convex, plotGeneratedLines, gen_abs_uniform, fusePairs
from Symmetries.WernerSysSwapSymmetry_p2 import WernerSysSwapSymmetry_p1, WernerSysSwapSymmetry_p2
from WernerESSQNN import dev1, dev2, dev1_p2, dev2_p2, enc_func, ansatz, enc_func_p2, ansatz_p2, \
    enc_func_a2, ansatz_p2_ps_c2, ansatz_p2_ps_c1, ansatz_p2_ps_minimum

# ! ---------------------------------- Control Panel ----------------------------------
# data
SAMPLE_STRATEGY = "uniform2"  # "uniform" or "randEachArea" or "uniform2"
useShuffleSeed = False  # only for uniform samples (not include uniform2)
SHUFFLE_SEED = 42  # 42
# model
p2FlagOn = True  # use paralleled encoding or not (including zero paralleled)
enc_func = enc_func_a2  # nop: enc_func (pure enc) p2: enc_func_p2 (parallel encoded) enc_func_a2 (zero paralleled)
ansatz = ansatz_p2_ps_c1  # ansatz_p2_ps_minimum ansatz_p2_ps_c1 ansatz_p2_ps_c2 (three for complexing cced) ansatz_p2 ansatz
# ini params
NUM_PARAMS_EACH = 72
params_taped = qnp.random.uniform(0, 2 * qnp.pi, size=(2, NUM_PARAMS_EACH), requires_grad=True)
# train
NUM_EPOCH = 50
LAMBDA = 1
STEP_SIZE = 0.4
opt = qml.AdamOptimizer(stepsize=STEP_SIZE)  # GradientDescentOptimizer
# checking
CHECK_INTERVAL = 1  # interval of epochs for printing training info
plotFlag = "best"  # "best" "final" "off"

recordedPrints = []

def savedPrint(*args, **kwargs):
    recordedPrints.append(' '.join(map(str, args)))
    print(*args, **kwargs)

NUM_QUBITS_SUBSTM = 1
NUM_ANCILLA = 2  # for ancilla case, case under parallel should be further considered

# ! ---------------------------------- Data ----------------------------------
if SAMPLE_STRATEGY == "uniform":
    NUM_SAMPLES_LINE = 5
    lodaPathRoot = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE}"
    abPairs = gen_abs_uniform(NUM_SAMPLES_LINE)
    if NUM_SAMPLES_LINE == 5:  # total
        INDEX_SPLIT, BATCH_SIZE = 12, 5  #
    elif NUM_SAMPLES_LINE == 11:
        INDEX_SPLIT, BATCH_SIZE = 55, 11
    # for BATCH_SIZE 5->3, 11->11 (also could be just use NUM_SAMPLES_LINE, which could be divided by this number)
    # if useShuffleSeed:
    #     np.random.seed(SHUFFLE_SEED)
    # np.random.shuffle(ds)
elif SAMPLE_STRATEGY == "uniform2":
    NUM_SAMPLES_LINE = 11
    NUM_SAMPLES_LINE2 = 5
    lodaPathRoot = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE}_{NUM_SAMPLES_LINE2}"
    abPairs = fusePairs(gen_abs_uniform(NUM_SAMPLES_LINE), gen_abs_uniform(NUM_SAMPLES_LINE2))
    if NUM_SAMPLES_LINE == 11 and NUM_SAMPLES_LINE2 == 5:
        INDEX_SPLIT, BATCH_SIZE = 60, 10
elif SAMPLE_STRATEGY == "randEachArea":
    NUM_EACH_AREA_TRAIN = 10
    NUM_EACH_AREA_TEST = 2
    lodaPathRoot = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_EACH_AREA_TRAIN}_{NUM_EACH_AREA_TEST}"
    abPairs = np.load(lodaPathRoot + "_abPairs.npy", allow_pickle=True)
    INDEX_SPLIT, BATCH_SIZE = NUM_EACH_AREA_TRAIN * 3, 5  # total samples = 3 * (NUM_EACH_AREA_TRAIN + NUM_EACH_AREA_TEST)

# * load data
ds = np.load(lodaPathRoot + ".npy", allow_pickle=True)
# * train test split
train_ds, test_ds = ds[:INDEX_SPLIT], ds[INDEX_SPLIT:]
# * Batch
NUM_BATCH = len(train_ds) // BATCH_SIZE

# ! ---------------------------------- QNN and Symmetry ----------------------------------
# Define the quantum devices and QNodes
if not p2FlagOn:
    dev1, dev2 = dev1, dev2
    # ob1, ob2 = SWAP, SWAPP
    ob1, ob2 = np.kron(Id, Z), np.kron(Id, Z)
    # ob1, ob2 = np.kron(Z, Z), np.kron(Z, Z) # 0.75 on train_ds, 1.0 on test_ds
    O1, O2 = qml.Hermitian(ob1, wires=[0, 1]), qml.Hermitian(ob2, wires=[0, 1])
    NUM_ANCILLA = 0
else:
    dev1, dev2 = dev1_p2, dev2_p2
    # ob1, ob2 = tensorOfListOps2(Id, Id, Id, Z), tensorOfListOps2(Id, Id, Id, Z)
    ob1, ob2 = tensorOfListOps2(Id, Id, Id, X), tensorOfListOps2(Id, Id, Id, X)
    # ob1, ob2 = tensorOfListOps2(SWAP, SWAP), tensorOfListOps2(SWAPP, SWAPP)
    O1, O2 = qml.Hermitian(ob1, wires=range(4)), qml.Hermitian(ob2, wires=range(4))

qnn1 = EasyPQC(dev1, enc_func, ansatz, O1, O1)
qnn2 = EasyPQC(dev2, enc_func, ansatz, O2, O2)

# the combined model
def model(parameters, rho):
    tr1 = qnn1.circuit(rho, parameters[0])
    tr2 = qnn2.circuit(rho, parameters[1])
    return sigmoid(tr1) * sigmoid(tr2)
    # return 1 if (tr1 >= 0 and tr2 >= 0) else 0 # 1 is separable, 0 is entangled

# ! ---------------------------------- Train and log ----------------------------------

# * ini of params and record
paramsG_taped = copy.deepcopy(params_taped)
bestTrainAcc, bestTrainAccG, bestTrainEpoch, bestTrainEpochG = 0, 0, 0, 0
bestTrainParams, bestTrainParamsG = params_taped, paramsG_taped

savedPrint("Before training")
savedPrint("Accuracy on train_ds: ", accuracy(qnn1, qnn2, params_taped, train_ds))
savedPrint("Accuracy on test_ds: ", accuracy(qnn1, qnn2, params_taped, test_ds))

savedPrint("---------------- Start training ----------------")
for i in range(NUM_EPOCH):
    # * batched train
    np.random.shuffle(train_ds)
    for j in range(0, NUM_BATCH):
        batch = train_ds[j * BATCH_SIZE:j * BATCH_SIZE + BATCH_SIZE]
        params_taped = opt.step(lambda v: cost(model, v, batch), params_taped)
        paramsG_taped = opt.step(lambda v: costG(qnn1, qnn2, model, v, NUM_QUBITS_SUBSTM, NUM_ANCILLA, LAMBDA, batch), paramsG_taped)
    # * metric values
    cst = cost(model, params_taped, train_ds)
    cstG = costG(qnn1, qnn2, model, paramsG_taped, NUM_QUBITS_SUBSTM, NUM_ANCILLA, LAMBDA, train_ds)
    acc = accuracy(qnn1, qnn2, params_taped, train_ds)
    accG = accuracy(qnn1, qnn2, paramsG_taped, train_ds)
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
        if accG > bestTrainAccG:
            bestTrainAccG = accG
            bestTrainEpochG = i
            bestTrainParamsG = paramsG_taped
    # * log
    if (i + 1) % CHECK_INTERVAL == 0 or i < 10:
        savedPrint(f"Eopch: {i + 1:02}, train_ds cst: {cst :0.4f}:, acc: {acc: 0.4f}, cstG: {cstG :0.4f}:, accG: {accG: 0.4f}")
savedPrint("---------------- Finished training ----------------")

# * print final result
acc_train, acc_test = accuracy(qnn1, qnn2, params_taped, train_ds), accuracy(qnn1, qnn2, params_taped, test_ds)
accG_train, accG_test = accuracy(qnn1, qnn2, paramsG_taped, train_ds), accuracy(qnn1, qnn2, paramsG_taped, test_ds)
savedPrint(f"FinalTrain:\nacc_train: {acc_train: 0.4f}, acc_test: {acc_test: 0.4f}\naccG_train: {accG_train: 0.4f}, accG_test: {accG_test: 0.4f}")
# * print best result
bestTrainAccTest, bestTrainAccGTest = accuracy(qnn1, qnn2, bestTrainParams, test_ds), accuracy(qnn1, qnn2, bestTrainParamsG, test_ds)
savedPrint(f"bestTrain:\nbestTrainAcc: {bestTrainAcc: 0.4f}, bestTrainAccTest: {bestTrainAccTest: 0.4f}, at Epoch: {bestTrainEpoch+1}\n\
bestTrainAccG: {bestTrainAccG: 0.4f}, bestTrainAccGTest: {bestTrainAccGTest: 0.4f}, at Epoch: {bestTrainEpochG+1}")
print("\nMain Finished.")


saveDate = datetime.datetime.now().strftime("%m%d%H%M")
# ! ---------------------------------- Plot ----------------------------------
if plotFlag == "best":
    fig, ax = plotGeneratedLines(dm_convex, ansatz, ansatz, bestTrainParams[0], bestTrainParams[1], ob1, ob2, abPairs)
    fig.suptitle(f"ESS {saveDate} {SAMPLE_STRATEGY} {ansatz.__name__}: NoG")
    plt.show()
    fig, ax = plotGeneratedLines(dm_convex, ansatz, ansatz, bestTrainParamsG[0], bestTrainParamsG[1], ob1, ob2, abPairs)
    fig.suptitle(f"ESS {saveDate} {SAMPLE_STRATEGY} {ansatz.__name__}: G")
    plt.show()
elif plotFlag == "final":
    fig, ax = plotGeneratedLines(dm_convex, ansatz, ansatz, params_taped[0], params_taped[1], ob1, ob2, abPairs)
    fig.suptitle(f"ESS {saveDate} {SAMPLE_STRATEGY} {ansatz.__name__}: NoG")
    plt.show()
    fig, ax = plotGeneratedLines(dm_convex, ansatz, ansatz, paramsG_taped[0], paramsG_taped[1], ob1, ob2, abPairs)
    fig.suptitle(f"ESS {saveDate} {SAMPLE_STRATEGY} {ansatz.__name__}: G")
    plt.show()

# ! ---------------------------------- Save ----------------------------------
# * save settings
savePathRoot = f"./Rs/ESS_{saveDate}_{SAMPLE_STRATEGY}_{ansatz.__name__}"
# * save params
np.save(savePathRoot + "_params.npy", np.concatenate((params_taped, paramsG_taped, bestTrainParams, bestTrainParamsG)))
# * save settings and logs
with open(savePathRoot + "_log.txt", "w") as f:
    f.write(f"saveDate: {saveDate}\n\n")
    f.write(f"SampleStrategy: {SAMPLE_STRATEGY}\n")
    f.write(f"NUM_SAMPLES_LINE: {NUM_SAMPLES_LINE}\n")
    if SAMPLE_STRATEGY == "uniform2":
        f.write(f"NUM_SAMPLES_LINE2: {NUM_SAMPLES_LINE2}\n")
    f.write(f"useShuffleSeed: {useShuffleSeed}\nShuffleSeed: {SHUFFLE_SEED}\n")
    f.write(f"INDEX_SPLIT: {INDEX_SPLIT}\n")
    f.write(f"Ansatz: {ansatz.__name__}\n")
    f.write(f"\nNUM_BATCH: {NUM_BATCH}\n")
    f.write(f"Num_Epoch: {NUM_EPOCH}\n")
    f.write(f"Step_Size: {STEP_SIZE}\n")
    f.write(f"SG_Weight: {LAMBDA}\n")
    f.write(f"\nLogs:\n")
    for line in recordedPrints:
        f.write(line + "\n")
    f.write("\nOther Info:\n")
    f.write(f"Ob1: \n{ob1}\nOb2: \n{ob2}\n")
print("Plot and Save Finished.")
# ! ---------------------------------- Test applying sym, but the sym was wrong when below codes are used ----------------------------------

# from Utils import X, Y, dagger
# from Symmetries.WernerSysSwapSymmetry import WernerSysSwapSymmetry
# from tabulate import tabulate
#
# symmetry = WernerSysSwapSymmetry(num_qubit=1)
# test_sym_op = np.kron(X, X)
#
# headers = ['Index', 'tr1', 'tr1_sym', 'tr2', 'tr2_sym', 'sigmoid(tr1)', 'sigmoid(tr2)', 'tr pred.', 'tr_sym pred.', 'label']
# data = []
#
# def prediction(tr1, tr2):
#     return 1 if (tr1 > 0 and tr2 > 0) else 0
#
# for idx, (rho, label) in enumerate(ds):
#     tr1 = qnn1(rho, params[0:4])
#     tr2 = qnn2(rho, params[4:])
#     # tr1_sym = qnn1(test_sym_op @ rho @ dagger(test_sym_op), params[0:4])
#     # tr2_sym = qnn2(test_sym_op @ rho @ dagger(test_sym_op), params[4:])
#     tr1_sym = qnn1(symmetry._twirling(rho), params[0:4])
#     tr2_sym = qnn2(symmetry._twirling(rho), params[4:])
#
#     data.append((idx, tr1, tr1_sym, tr2, tr2_sym, sigmoid(tr1), sigmoid(tr2), prediction(tr1, tr2), prediction(tr1_sym, tr2_sym), label))
#
# print(tabulate(data, headers, tablefmt='pretty'))
