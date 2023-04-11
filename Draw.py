import matplotlib.pyplot as plt
from itertools import product
import pennylane as qml
import pennylane.numpy as np
import pickle

num_bits = 6
NUM_LAYER = 28
with open('parameter 4-10 2.pkl', 'rb') as f:
    record = pickle.load(f)
    para, paraG= record[16]
    # paraG= record[16]

dev = qml.device('default.qubit', wires=num_bits)

#%% Ansatz
def ansatz(parameters, num_layers):
    para_arr = [parameters[i:i+num_bits] for i in range(0, len(parameters), num_bits)]
    para_arr.reverse()

    qml.broadcast(qml.RZ, wires=range(num_bits), 
                  pattern="single", parameters=para_arr.pop() )
    for i in range(num_layers):
        pattern = "double" if i%2 == 0 else "double_odd"
        layer(num_bits, para_arr.pop(), pattern)
    qml.broadcast(qml.RX, wires=range(num_bits), 
                  pattern="single", parameters=para_arr.pop() )
    
@qml.qnode(dev)
def circuit(x, parameters, num_layers):
    # x = np.append(x,x)
    # x = np.append(x,[0])
    qml.broadcast(qml.RY, wires=range(num_bits), 
                  pattern="single", parameters=x )
    ansatz(parameters, num_layers)

    return qml.expval(qml.PauliZ(num_bits-1))


def layer(num_bits, parameters, pattern):
    qml.broadcast(qml.RY, wires=range(num_bits), 
                  pattern="single", parameters=parameters )
    qml.broadcast(qml.CNOT, wires=range(num_bits), pattern=pattern)

plt.style.use('_mpl-gallery-nogrid')

bound = np.pi * 0.9
# make data
X, Y = np.meshgrid(
    np.linspace(-bound, bound, 16), np.linspace(-bound, bound, 16)
    )
Z = np.zeros_like(X)
for i,j in product(range(len(X)), range(len(X))):
    data = [X[i,j], Y[i,j]]*3
    Z[i,j] = circuit(data, paraG, NUM_LAYER)
levels = np.linspace(Z.min(), Z.max(), 11)

# plot
fig, ax = plt.subplots()

ax.contourf(X, Y, Z, levels=levels)

plt.show()