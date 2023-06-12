import matplotlib.pyplot as plt
from matplotlib import cm
import pennylane as qml
import numpy as np
import pickle
with open('res 2b (0).pkl', 'rb') as f:
    [cst_lst, cstG_lst, para_lst, paraG_lst] = pickle.load(f)
plt.style.use('_mpl-gallery')

# Make data
X = np.arange(0, np.pi, 0.2)
Y = np.arange(0, np.pi, 0.2)
X, Y = np.meshgrid(X, Y)

cst_land = np.zeros_like(X)
sg_land = np.zeros_like(X)
from DataGener import SysSwapSymmetry
from Train_2b import cost, symmetry, circuit
from itertools import product
for i,j in product(range(len(X)), repeat=2):
    cst_land[i,j] = cost([X[i,j], Y[i,j]])
    U = qml.matrix(circuit)([X[i,j], Y[i,j]])
    sg_land[i,j] = symmetry.symmetry_guidance(U) * 1.5

# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y, cst_land, vmin=cst_land.min() * 2, cmap=cm.Blues)
# ax.plot_surface(X, Y, sg_land, vmin=sg_land.min() * 2, cmap=cm.Reds)
ax.plot_surface(X, Y, cst_land, cmap=cm.Blues, label='cost')
ax.plot_surface(X, Y, sg_land,  cmap=cm.Reds, label='SG')
# ax.legend()
ax.set_title('The landscape of model')

# ax.set(xticklabels=[],
#        yticklabels=[],
#        zticklabels=[])

plt.show()

fig.savefig('2bits landscape.png')