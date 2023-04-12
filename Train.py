#%% import
import pennylane.numpy as np
import pennylane as qml
import pickle, copy
from DataGenerator import SwapSymmetry
with open('Datarepo/data_symmetry4.pkl', 'rb') as f:
        [num_bits, train_data, test_data, symmetry] = pickle.load(f)


dev = qml.device('default.mixed', wires=num_bits)

#%% Ansatz
def ansatz(parameters, num_layers):
    '''PQC w/o input encoding'''
    para_arr = [parameters[i:i+num_bits] for i in range(0, len(parameters), num_bits)]
    para_arr.reverse()

    qml.broadcast(qml.RZ, wires=range(num_bits), 
                  pattern="single", parameters=para_arr.pop() )
    for i in range(num_layers):
        pattern = "double" if i%2 == 0 else "double_odd"
        layer(num_bits, para_arr.pop(), pattern)
    qml.broadcast(qml.RX, wires=range(num_bits), 
                  pattern="single", parameters=para_arr.pop() )
    
@qml.qnode(dev, diff_method="backprop", interface="torch")
def circuit(x, parameters, num_layers):
    '''whole circuit'''
    # x = np.append(x,x)
    # x = np.append(x,[0])
    qml.broadcast(qml.RY, wires=range(num_bits), 
                  pattern="single", parameters=x )
    ansatz(parameters, num_layers)

    return qml.expval(qml.PauliZ(num_bits-1))

def layer(num_bits, parameters, pattern):
    '''layer-build helper function for ansatz construction'''
    qml.broadcast(qml.RY, wires=range(num_bits), 
                  pattern="single", parameters=parameters )
    qml.broadcast(qml.CNOT, wires=range(num_bits), pattern=pattern)

# %% cost function

def cost(parameter, num_layers):
    '''cost function without symmetry guidance'''
    # cst_values =  [(circuit(x, parameter, num_layers) + (-1)**label)**2
    #                 for x, label in train_data ]
    cst_values = []
    for x, label in train_data:
        cst_values.append(
            ( circuit(x, parameter, num_layers) + (-1)**label )**2
            )
    return sum(cst_values) / len(train_data)

@qml.qnode(dev, diff_method="backprop", interface="torch")
def U_circ(parameters, num_layer):
    """turning the ansatz into a unitary matrix"""
    ansatz(parameters, num_layer)
    return qml.expval(qml.Identity(0))

def costG(parameter, num_layer, lamd):
    '''cost function with symmetry guidance'''
    U = qml.matrix(U_circ)(parameter, num_layer)
    cst = cost(parameter, num_layer)
    g = symmetry.symmetry_guidance(U) 
    return cst + lamd * g

def accuracy(parameter, num_layer):
    correct = 0
    for rho,label in test_data: 
        prediction = np.ceil(circuit(rho, parameter, num_layer))
        if np.abs(prediction - label) < 1 : correct += 1
    return correct / len(test_data)



from functools import partial

def run(record, NUM_LAYER):
    
    ETA = 0.04
    LAMD = 0.5
    MAX_ITER = 30
    NUM_PARA = (NUM_LAYER + 2) * num_bits
    opt = qml.AdamOptimizer(stepsize = ETA)

    cst_lst = []
    cstG_lst = []
    # winner = []
    para_init = np.random.random(NUM_PARA, requires_grad=True) *2*np.pi

    cost_ = partial(cost, num_layers = NUM_LAYER)
    costG_ = partial(costG, num_layer = NUM_LAYER, lamd = LAMD)
    accuracy_ = partial(accuracy, num_layer = NUM_LAYER)
    para = para_init
    
    #* shared pre-training
    CUT = int(MAX_ITER/2)
    for it in range(CUT):
        para = opt.step(cost_,para)
        print('\r', f"pre-training {(it+1)/CUT : .2%}", end='', flush=True)
    
    print('\n')
    paraG = copy.deepcopy(para)

    #* training for comparison
    for it in range(CUT):

        para, cst = opt.step_and_cost(cost_, para)
        paraG, cstG = opt.step_and_cost(costG_, paraG)
        cst_lst.append(cst)
        cstG_lst.append(cstG/2)
        # ac, acG = accuracy_(para), accuracy_(paraG)
        record.append((para,paraG))

        if it % 4 == 0:
            print( f"Iter: {it + 1 : 3d} | Cost: {cst : 0.4f} | CostG: {cstG : 0.4f} | accuracy: {accuracy_(para) : 3.2%} | accuracyG: {accuracy_(paraG) : 3.2%}" )
# %%
if __name__ == '__main__':
    from time import time
    record = []
    NUM_LAYER = 28

    t0 = time()
    run(record, NUM_LAYER)
    t1 = time()
    print(f"Run time: {(t1-t0)/60 : .2f} min")
    

    # import matplotlib.pyplot as plt
    # from itertools import product

    # plt.style.use('_mpl-gallery-nogrid')

    # bound = np.pi * 0.9
    # # make data
    # X, Y = np.meshgrid(
    #     np.linspace(-bound, bound, 16), np.linspace(-bound, bound, 16)
    #     )
    # Z = np.zeros_like(X)
    # for i,j in product(range(len(X)), range(len(X))):
    #     data = [X[i,j], Y[i,j]]*3
    #     Z[i,j] = circuit(data, record[-1][0], NUM_LAYER)
    # levels = np.linspace(Z.min(), Z.max(), 11)

    # # plot
    # fig, ax = plt.subplots()

    # ax.contourf(X, Y, Z, levels=levels)

    # plt.show()
    # %%
