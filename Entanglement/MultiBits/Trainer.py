import numpy as np
from time import time
from itertools import product
from functools import partial
import matplotlib.pyplot as plt


from QiskitUtils.MyQC import QNN_for_Werner
from QiskitUtils.TrainTool import Optimizer
from QiskitUtils.Losses import cost, costG, accuracy


def train1(CONFIG, sample_set, symmetry, pre_para=[]):
    """Qiskit training 1"""

    NUM_PART, NUM_LAYER, MAX_ITER, ETA, LAMBDA = CONFIG['num_part'], CONFIG['num_layer'], CONFIG['max_iter'], CONFIG['eta'], CONFIG['lambda']


    qnn = QNN_for_Werner(NUM_PART, NUM_LAYER, 0)
    qnnG = QNN_for_Werner(NUM_PART, NUM_LAYER, 0)
    init_para = qnn.rand_para()
    if pre_para:
        init_para[:len(pre_para[-1])] = pre_para[-1]

    loss = partial(cost, qnn=qnn, dataSet=sample_set)
    opt = Optimizer(loss, init_para, MAX_ITER, ETA)
    lossG = partial(costG, qnn=qnnG, dataSet=sample_set, symmetry=symmetry, lamd=LAMBDA)
    optG = Optimizer(lossG, init_para, MAX_ITER, ETA)

    t0 = time()
    y1 = opt.gradient_descent()
    y2 = optG.gradient_descent()
    print(f"the optimal cost without SG is {y1 : .2f}")
    print(f"the optimal cost with SG is {y2 : .2f}")
    delta = optG.y_lst[0] - opt.y_lst[0]
    plt.plot(opt.y_lst, label = 'without SG')
    plt.plot(np.array(optG.y_lst) - delta, label = 'with SG')
    print(f"Accuracy in test set without SG is {accuracy(opt.x_lst[-1]) : 3.2%}")
    print(f"Accuracy in test set with SG is {accuracy(optG.x_lst[-1]) : 3.2%}")

    t1 = time()

    print( f'\nThe runtime is  {(t1-t0)/60 : .2f} min' )
    plt.legend()
    plt.show()

#! ------------------------------- train and Compare ------------------------------- #

from Entanglement.MultiBits.DrawHelper import Result

def train2(num_layer, num_iter, dataSet, symmetry, NUM_PART, ETA, LAMBDA):
    """Qiskit training 2, which is called by the function defined below"""

    #* models
    qnn = QNN_for_Werner(NUM_PART, num_layer=num_layer, num_ancilla=0)
    qnnG = QNN_for_Werner(NUM_PART, num_layer=num_layer, num_ancilla=0)
    init_para = qnn.rand_para()

    #* losses
    loss = partial(cost, qnn=qnn, dataSet=dataSet)
    lossG = partial(costG, qnn=qnnG, dataSet=dataSet, symmetry=symmetry, lamd=LAMBDA)
    
    #* training
    opt = Optimizer(loss, init_para, maxiter=num_iter, eta = ETA)
    optG = Optimizer(lossG, init_para, maxiter=num_iter, eta = ETA)
    opt.gradient_descent()
    optG.gradient_descent()

    #* results
    ac_drt = accuracy(opt.x_lst[-1])
    ac_gdc = accuracy(optG.x_lst[-1])
    direct_res = Result(opt.x_lst, opt.y_lst, opt.x_lst[-1], ac_drt)
    guid_res = Result(optG.x_lst, optG.y_lst, optG.x_lst[-1], ac_gdc)

    #* winner
    if ac_gdc > ac_drt: winner = 1
    elif ac_gdc < ac_drt: winner = -1
    else: winner = 0

    return direct_res, guid_res, winner


def iterCompare(layer_list, iter_list, CONFIG, dataSet, symmetry):
    """for comparing model performance under sg and w/o sg by iterating layers and max_iters"""

    NUM_PART, ETA, LAMBDA = CONFIG['num_part'], CONFIG['eta'], CONFIG['lambda']

    res = []
    for num_layer, num_iter in product(layer_list, iter_list):
        res.append((num_layer, num_iter ,*train2(num_layer, num_iter, dataSet, symmetry, NUM_PART, ETA, LAMBDA)))
    return res