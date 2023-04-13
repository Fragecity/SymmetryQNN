import pennylane as qml
import copy
from functools import partial
from Losses import cost, costG, accuracy
from Ansatzs.PQCBase import PQCBase


def trainAndCompare(record, pqc: PQCBase, para_init, symmetry, train_data, test_data, ETA, LAMD, MAX_ITER):
    
    opt = qml.AdamOptimizer(stepsize=ETA)

    cst_lst = []
    cstG_lst = []
    
    cost_ = partial(cost, pqc=pqc, train_data=train_data)
    costG_ = partial(costG, pqc=pqc, symmetry=symmetry, lamd=LAMD, train_data=train_data)
    accuracy_ = partial(accuracy, pqc=pqc, test_data=test_data)
    
    para = para_init

    # * shared pre-training
    CUT = int(MAX_ITER / 2)
    for it in range(CUT):
        para = opt.step(cost_, para)
        print('\r', f"pre-training {(it + 1) / CUT : .2%}", end='', flush=True)

    print('\n')
    paraG = copy.deepcopy(para)

    # * training for comparison
    for it in range(CUT):   # CUT MAX_ITER

        para, cst = opt.step_and_cost(cost_, para)
        paraG, cstG = opt.step_and_cost(costG_, paraG)
        cst_lst.append(cst)
        cstG_lst.append(cstG / 2)
        # ac, acG = accuracy_(para), accuracy_(paraG)
        record.append((para, paraG))

        if it % 4 == 0:
            print(
                f"Iter: {it + 1 : 3d} | Cost: {cst : 0.4f} | CostG: {cstG : 0.4f} | accuracy: {accuracy_(para) : 3.2%} | accuracyG: {accuracy_(paraG) : 3.2%}")
