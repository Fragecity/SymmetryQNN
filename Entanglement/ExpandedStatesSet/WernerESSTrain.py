import numpy as np
import copy
import pennylane as qml
from pennylane import numpy as qnp
from Utils import SWAP, SWAPP, Id, X, Z, tensorOfListOps2
from losses import cost, costG, accuracy
from ESSUtils import sigmoid
from Ansatzs.EasyPQC import EasyPQC
from Symmetries.WernerSysSwapSymmetry_p2 import WernerSysSwapSymmetry_p1, WernerSysSwapSymmetry_p2

#! ---------------------------------- Data ----------------------------------
#* data loading
SAMPLE_STRATEGY = "uniform"
NUM_SAMPLES_LINE = 5
lodaPath = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE}.npy"
ds = np.load(lodaPath, allow_pickle=True)

# np.random.seed(42)
np.random.shuffle(ds)
INDEX_SPLIT = 12 # 5->12, 11->55
train_ds, test_ds = ds[:INDEX_SPLIT], ds[INDEX_SPLIT:]

BATCH_SIZE = 5 # 5->3, 11->11 (also could be just use NUM_SAMPLES_LINE, which could be divided by this number)
NUM_BATCH = len(train_ds)//BATCH_SIZE

#! ---------------------------------- Train Settings ----------------------------------

NUM_EPOCH = 100
STEP_SIZE = 0.4

NUM_PARAMS_EACH = 72
# Initialize parameters
params_taped = qnp.random.uniform(0, 2*qnp.pi, size=(2, NUM_PARAMS_EACH), requires_grad=True)
# known_params = qnp.array([qnp.pi/4, -qnp.pi/2, 0, qnp.pi/2, -qnp.pi, 1.8026, qnp.pi*3/4, 0, qnp.pi/2, 1.3453, 3.0874, -1.3328, qnp.pi, 0.40796, 1.9788, 1.339, -qnp.pi, -qnp.pi/2])
# params_taped[0, :len(known_params)] = known_params
# params_taped[0, len(known_params):] = known_params
# params_taped[1, :len(known_params)] = known_params
# params_taped[1, len(known_params):] = known_params
# print("Initial params: ", params[0])

# Optimizer
# opt = qml.GradientDescentOptimizer(stepsize= 1)
opt = qml.AdamOptimizer(stepsize=STEP_SIZE)

#! ---------------------------------- QNN and Symmetry ----------------------------------
from WernerESSQNN import dev1, dev2, dev1_p2, dev2_p2, enc_func, ansatz, enc_func_p2, ansatz_p2, \
    enc_func_a2, ansatz_p2_ps_c2, ansatz_p2_ps_c1, ansatz_p2_ps_minimum
# Define the quantum devices and QNodes

p2Flag = True
# p2Flag = False

if p2Flag == False:
    dev1, dev2 = dev1, dev2
    # ob1, ob2 = SWAP, SWAPP
    ob1, ob2 = np.kron(Id, Z), np.kron(Id, Z)
    # ob1, ob2 = np.kron(Z, Z), np.kron(Z, Z) # 0.75 on train_ds, 1.0 on test_ds
    O1, O2 = qml.Hermitian(ob1, wires=[0,1]), qml.Hermitian(ob2, wires=[0,1])
    symmetry1 = WernerSysSwapSymmetry_p1(num_qubit=1, global_ob=ob1)
    symmetry2 = WernerSysSwapSymmetry_p1(num_qubit=1, global_ob=ob2)
    enc_func = enc_func
    ansatz = ansatz
else:
    dev1, dev2 = dev1_p2, dev2_p2
    # ob1, ob2 = tensorOfListOps2(Z, Z, Z, Z), tensorOfListOps2(Z, Z, Z, Z)
    # ob1, ob2 = tensorOfListOps2(Id, Id, Id, Z), tensorOfListOps2(Id, Id, Id, Z)
    ob1, ob2 = tensorOfListOps2(Id, Id, Id, X), tensorOfListOps2(Id, Id, Id, X)
    # ob1, ob2 = tensorOfListOps2(SWAP, SWAP), tensorOfListOps2(SWAPP, SWAPP)
    O1, O2 = qml.Hermitian(ob1, wires=range(4)), qml.Hermitian(ob2, wires=range(4))
    symmetry1 = WernerSysSwapSymmetry_p2(num_qubit=1, global_ob=ob1)
    symmetry2 = WernerSysSwapSymmetry_p2(num_qubit=1, global_ob=ob2)
    # enc_func = enc_func_p2
    enc_func = enc_func_a2
    # ansatz = ansatz_p2
    ansatz = ansatz_p2_ps_c2

LAMBDA = 1

qnn1 = EasyPQC(dev1, enc_func, ansatz, O1, O1)
qnn2 = EasyPQC(dev2, enc_func, ansatz, O2, O2)

# the combined model
def model(parameters, rho):
    tr1 = qnn1.circuit(rho, parameters[0])
    tr2 = qnn2.circuit(rho, parameters[1])
    return sigmoid(tr1)*sigmoid(tr2)
    # return 1 if (tr1 > 0 and tr2 > 0) else 0

#! ---------------------------------- Training ----------------------------------

paramsG_taped = copy.deepcopy(params_taped)

print("Before training")
print("Accuracy on train_ds: ", accuracy(qnn1, qnn2, params_taped, train_ds))
print("Accuracy on test_ds: ", accuracy(qnn1, qnn2, params_taped, test_ds))

print("Starting training...")
for i in range(NUM_EPOCH):
    for j in range(0, NUM_BATCH):
        batch = train_ds[j*BATCH_SIZE:j*BATCH_SIZE+BATCH_SIZE]
        batchtmp = batch
        params_taped = opt.step(lambda v: cost(model, v, batch), params_taped)
        paramsG_taped = opt.step(lambda v: costG(qnn1, qnn2, model, v, symmetry1, symmetry2, LAMBDA, batchtmp), paramsG_taped)
    if (i+1) % 5 == 0 or i < 10:
        cst = cost(model, params_taped, train_ds)
        cstG = costG(qnn1, qnn2, model, paramsG_taped, symmetry1, symmetry2, LAMBDA, train_ds)
        acc = accuracy(qnn1, qnn2, params_taped, train_ds)
        accG = accuracy(qnn1, qnn2, paramsG_taped, train_ds)
        print(f"Eopch: {i+1:02}, train_ds cst: {cst :0.4f}:, acc: {acc: 0.4f}, cstG: {cstG :0.4f}:, accG: {accG: 0.4f}")

print("Finished training")

acc_train = accuracy(qnn1, qnn2, params_taped, train_ds)
acc_test = accuracy(qnn1, qnn2, params_taped, test_ds)
accG_train = accuracy(qnn1, qnn2, paramsG_taped, train_ds)
accG_test = accuracy(qnn1, qnn2, paramsG_taped, test_ds)
print(f"acc_train: {acc_train: 0.4f}, acc_test: {acc_test: 0.4f}\n accG_train: {accG_train: 0.4f}, accG_test: {accG_test: 0.4f}")

#! ---------------------------------- Save ----------------------------------

#* save
# np.save("./TmpRecord/ESS_params_070701.npy", params)

#! ---------------------------------- Test ----------------------------------

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



#! ---------------------------------- QNN Deprecated codeForm ----------------------------------

# @qml.qnode(dev1, diff_method="backprop", interface="torch")
# def qnn1(rho, params):
#     qml.QubitDensityMatrix(rho, wires=[0, 1])
#     qml.RY(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.CNOT(wires=[0, 1])
#     qml.RY(params[2], wires=0)
#     qml.RY(params[3], wires=1)
#     return qml.expval(O1)
#
# @qml.qnode(dev2, diff_method="backprop", interface="torch")
# def qnn2(rho, params):
#     qml.QubitDensityMatrix(rho, wires=[0, 1])
#     qml.RY(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.CNOT(wires=[0, 1])
#     qml.RY(params[2], wires=0)
#     qml.RY(params[3], wires=1)
#     return qml.expval(O2)



