from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit
import numpy as np
from numpy import exp
from copy import copy
from functools import partial
from QiskitUtils.HelperFuncs import getUofCircuit

backend = AerSimulator()

def eval_expct(ansatz:QuantumCircuit, theta, shots = 1024):
    value_dict = dict(zip(ansatz.parameters, theta))
    qc = ansatz.bind_parameters(value_dict)
    job = backend.run(qc, shots = shots)
    counts = job.result().get_counts()
    temp = [(
            sum( [int(i) for i in key] )*(-2) + len(key)
            )*counts[key] for key in counts.keys() ] 
    return sum(temp)/shots


def logistic(x, k = 1):
    mother = 1 + exp(- k*x )
    return 1 / mother

def squeeze(x, bound = 2):
    return (x+bound)/bound/2

def grad(func, para: np.array, EPS = 0.05) -> np.array:
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

def grad_des(func, para, callback, maxiter = 100, eta = 0.5):
    """See also the below one"""
    theta = copy(para)
    rate = eta/maxiter
    for i in reversed(range(maxiter)):
        callback(func(theta), theta)
        theta = theta - i*rate * grad(func, theta)
        theta = np.mod(theta, 2*np.pi)

#*
# the one in multi qubits compare part
# def grad_des(func, para, callback, maxiter = 100, eta = 0.5):
#     theta = copy(para)
#     beg = int(0.3*maxiter)
#     end = int(1.3*maxiter)
#     rate = eta/end
#     for i in reversed(range(beg, end)):
#         callback(func(theta), theta)
#         theta = theta - i*rate * grad(func, theta)
#         theta = np.mod(theta, 2*np.pi)

def trainTwo(cost, qc, qnn_s, symmetry, MAX_ITER = 100, ETA = 0.5, LAMBDA = 1):
    """
    Training part of default and symmetry-guided training under qiskit DNS setting
    """
    #* w/o SG
    init_para = qnn_s.rand_paras()
    func_list = []
    para_list = []
    def callback(y,x):
        para_list.append(x)
        func_list.append(y)

    #* w/ SG
    funcG_list = []
    paraG_list = []

    U_cir = partial(getUofCircuit, qc=qc)

    def sg(x):
        return symmetry.symmetry_guidance(U_cir(x))

    def callback_g(y, x):
        funcG_list.append(y - LAMBDA*sg(x))
        paraG_list.append(x)

    #* Training
    grad_des(cost, init_para, callback, maxiter = MAX_ITER, eta=ETA) # w/o SG
    grad_des(lambda x: cost(x) + LAMBDA * sg(x), 
            init_para, callback_g, maxiter = MAX_ITER, eta = ETA) # w/ SG

    print("run over!")
    return para_list, func_list, paraG_list, funcG_list


class Optimizer():
    def __init__(self, object_fun, init_para, maxiter, eta) -> None:
        self.func = object_fun
        self.para = init_para
        self.maxiter = maxiter
        self.eta = eta
        self.x_lst = []
        self.y_lst = []
    
    def callback(self, y,x):
        self.y_lst.append(y)
        self.x_lst.append(x)
    
    def gradient_descent(self):
        grad_des(self.func, self.para, self.callback, self.maxiter, self.eta)
        # s = time.strftime("%Y-%m-%d %H-%M", time.localtime()) 
        # path = './data/result ' + self.func.__name__ +' ' + s + '.pkl'
        # with open(path, 'wb') as f:
        #     pickle.dump([self.x_lst, self.y_lst], f)
        # return self.y_lst[-1]
