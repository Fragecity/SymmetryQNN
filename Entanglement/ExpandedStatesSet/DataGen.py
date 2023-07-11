import sympy as sp
import numpy as np
from sympy import symbols, Matrix
from qiskit.quantum_info import DensityMatrix

from Utils import bellState
from Entanglement.WernerUtils import is_entangled
from Entanglement.ExpandedStatesSet.ESSUtils import gen_abs_uniform

#! ---------------------------------- Settings ----------------------------------
#* settings
SAMPLE_STRATEGY = "uniform"
NUM_SAMPLES_LINE = 11 # total number is (NUM_SAMPLES_LINE + 1) * NUM_SAMPLES_LINE / 2  5->15, 11->66
ds_savePath = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE}.npy"

#! ---------------------------------- Samples Plot and Preview ----------------------------------
#* samples plot
import matplotlib.pyplot as plt

#* generate
pairs = gen_abs_uniform(NUM_SAMPLES_LINE)
x, y = zip(*pairs)
plt.scatter(x, y)

for i, (xi, yi) in enumerate(pairs):
    plt.annotate(str(i), (xi, yi))

x = np.linspace(0, 1, 50)
y = 1/3*x + 1/3
plt.plot(x, y, color="r")
y = 3*x - 1
plt.plot(x, y, color="b")
plt.ylim([-0.1, 1.1])

plt.savefig(f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE}.png")
print("Samples plot saved.")
plt.show()

#! ---------------------------------- Gen Data ----------------------------------

#* constants
phi_plus = Matrix(bellState("phi_plus"))
phi_minus = Matrix(bellState("phi_minus"))

#* matrix form
a, b = symbols('a b')
dm_convex = (1-a-b) * sp.eye(4)/4 + a * phi_minus * phi_minus.H + b * phi_plus * phi_plus.H

#* uniform samples
ds = []
for a_val, b_val in gen_abs_uniform(NUM_SAMPLES_LINE):
    dmx = np.array(dm_convex.subs([(a, a_val), (b, b_val)]))
    dmx_dm = DensityMatrix(dmx)
    label = 0 if is_entangled(Matrix(dmx_dm.data), [2, 2]) else 1 # entangled as 0, separable as 1
    ds.append((dmx, label))

#! ---------------------------------- Save Data ----------------------------------
#* save
np.save(ds_savePath, ds)

#* notification
print("Data generation completed and saved.")