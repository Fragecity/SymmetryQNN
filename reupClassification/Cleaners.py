import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from losses import qnnPrediction

def optimizerSelector(OPTIMIZER, STEP_SIZE, **kwargs):
    if OPTIMIZER == 'adam':
        return qml.AdamOptimizer(stepsize=STEP_SIZE)  # GradientDescentOptimizer
    elif OPTIMIZER == 'gd':
        return qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
    else:
        raise RuntimeError("optimizer undefined")

def dsLoader(ds_type):
    if ds_type == 'basic' or ds_type == 'small_100':
        if ds_type == 'basic':
            data = pd.read_csv('trainSet_reUp.csv')
        elif ds_type == 'small_100':
            data = pd.read_csv('trainSet_reUp_small100.csv')
        X = data[['x', 'y']].values
        y = data['label'].values
        data_batch = list(zip(X, y))
        train_batch, test_batch = train_test_split(data_batch, test_size=0.2, random_state=42)
        return train_batch, test_batch

def ptInCircle(x, y, center=(0.5, 0.5), radius=0.2):
    '''Detect whether a samples in the circle (class 1/red) or not'''
    return np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius

def genDecisionArea(qnn_ob1, qnn_ob2, params, num_sample=100, title="", showPlot=False, ax=None, saveDMap=False):

    num_correct = 0
    samples = np.linspace(0, 1, num_sample)
    class1, class2 = [], []
    distributionMap = np.zeros((num_sample, num_sample))

    for idx_i, i in enumerate(samples):
        for idx_j, j in enumerate(samples):
            pred_val = qnnPrediction(qnn_ob1, qnn_ob2, params, [i,j])
            if pred_val == 1:
                class1.append([i, j])
                if ptInCircle(i, j): num_correct += 1
                distributionMap[idx_i, idx_j] = 1
            elif pred_val == -1:
                class2.append([i, j])
                if not ptInCircle(i, j): num_correct += 1
                distributionMap[idx_i, idx_j] = -1

    # overall accuracy
    acc_overall = num_correct/(num_sample**2)

    # gen plot then return the figure
    class1 = np.array(class1)
    class2 = np.array(class2)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # gen decision area
    ax.scatter(class1[:, 0], class1[:, 1], c='r', marker='s', label='class 1')
    ax.scatter(class2[:, 0], class2[:, 1], c='b', marker='o', label='class 2')
    ax.legend()
    ax.set_title(title + " acc_overall: {:.3f}".format(acc_overall))
    ax.set_aspect('equal')

    # plot the target circle
    circle = plt.Circle((0.5, 0.5), 0.2, color='k', fill=False)
    ax.add_artist(circle)

    if showPlot:
        plt.show()

    if ax is None:
        if saveDMap:
            return fig, distributionMap, acc_overall
        else:
            return fig
    else:
        if saveDMap:
            return distributionMap, acc_overall

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
    elif strategy == 'linearDec':
        return LAMBDA * ((NUM_EPOCH - epoch_at) / NUM_EPOCH)
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
