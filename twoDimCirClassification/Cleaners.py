import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

from losses import qnnPrediction

def optimizerSelector(OPTIMIZER, STEP_SIZE, **kwargs):
    if OPTIMIZER == 'adam':
        return qml.AdamOptimizer(stepsize=STEP_SIZE)  # GradientDescentOptimizer
    elif OPTIMIZER == 'gd':
        return qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
    else:
        raise RuntimeError("optimizer undefined")

def dsLoader(ds_type):
    if ds_type == '1500':
        return np.load("./Data/trainSet.npy", allow_pickle=True), np.load("./Data/testSet.npy", allow_pickle=True)
    elif ds_type == 'small':
        return np.load("./Data/trainSetSmall.npy", allow_pickle=True), np.load("./Data/testSetSmall.npy", allow_pickle=True)
    elif ds_type == 'small_tl':
        return np.load("./Data/trainSetSmall_tl.npy", allow_pickle=True), np.load("./Data/testSetSmall_tl.npy", allow_pickle=True)
    elif ds_type != '':
        return np.load("./Data/trainSet"+ds_type+".npy", allow_pickle=True), np.load("./Data/testSet"+ds_type+".npy",
                                                                                  allow_pickle=True)

# class tdDataSet:
#     def __init__(self, type, target, name):
#         self.data = data
#         self.data_path = ""
#         self.name = ""
#         self.batch_size = 0


def genDecisionArea(qnn, params, num_sample=100, title="", showPlot=False, ax=None):
    samples = np.linspace(0, 1, num_sample)
    class1, class2 = [], []
    for i in samples:
        for j in samples:
            if qnnPrediction(qnn, params, [i,j]) == 1:
                class1.append([i, j])
            elif qnnPrediction(qnn, params, [i,j]) == -1:
                class2.append([i, j])

    # gen plot then return the figure
    class1 = np.array(class1)
    class2 = np.array(class2)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.scatter(class1[:, 0], class1[:, 1], c='r', marker='s', label='class 1')
    ax.scatter(class2[:, 0], class2[:, 1], c='b', marker='o', label='class 2')
    ax.legend()
    ax.set_title(title)
    ax.set_aspect('equal')
    if showPlot:
        plt.show()

    if ax is None:
        return fig

def grad(func, para: np.array, EPS = 0.01) -> np.array: #0.05
    n = len(para)
    gradient = [0]*n
    II = np.identity(n)
    for i in range(n):
        # calculate partial f_i
        e_i = II[:, i]
        plus = para + EPS*e_i
        minus = para - EPS*e_i
        gradient[i] = (func(plus) - func(minus)) / (2*EPS)
    return np.array(gradient)

def optmizedParam(func, para, lr):
    para = para - lr * grad(func, para)
    para = np.mod(para, 2*np.pi)
    return para

def sgwScheduler(strategy, epoch_at, NUM_EPOCH, LAMBDA):
    if strategy == 'constant':
        return LAMBDA
    elif strategy == 'linear1':
        return LAMBDA * (epoch_at / NUM_EPOCH)
    elif strategy == 'postLinear3d4':
        if epoch_at <= 3/4 * NUM_EPOCH:
            return 0
        else:
            return 4*(NUM_EPOCH-epoch_at)/NUM_EPOCH * LAMBDA
    elif strategy == 'postLinear2d3':
        if epoch_at <= 2/3 * NUM_EPOCH:
            return 0
        else:
            return 3*(NUM_EPOCH-epoch_at)/NUM_EPOCH * LAMBDA
    else:
        raise RuntimeError("scheduler undefined")
