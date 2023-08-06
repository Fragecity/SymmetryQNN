import sympy as sp
import numpy as np
from sympy import symbols, Matrix
from qiskit.quantum_info import DensityMatrix
import matplotlib.pyplot as plt

from Utils import bellState
from Entanglement.WernerUtils import is_entangled
from Entanglement.ExpandedStatesSet.ESSUtils import gen_abs_uniform, genRandUniformABSamples, remove_duplicates_in_first

#! ---------------------------------- Settings ----------------------------------
#* settings
SAMPLE_STRATEGY = "uniform2" # uniform randEachArea uniform2

if SAMPLE_STRATEGY == "uniform":
    NUM_SAMPLES_LINE = 11 # total number is (NUM_SAMPLES_LINE + 1) * NUM_SAMPLES_LINE / 2  5->15, 11->66
    ds_savePathPre = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE}"
elif SAMPLE_STRATEGY == "uniform2":
    NUM_SAMPLES_LINE_1 = 11
    NUM_SAMPLES_LINE_2 = 5
    ds_savePathPre = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_SAMPLES_LINE_1}_{NUM_SAMPLES_LINE_2}"
elif SAMPLE_STRATEGY == "randEachArea":
    NUM_EACH_AREA_TRAIN = 10
    NUM_EACH_AREA_TEST = 2
    ds_savePathPre = f"./Data/{SAMPLE_STRATEGY}_samples_{NUM_EACH_AREA_TRAIN}_{NUM_EACH_AREA_TEST}"

#! ---------------------------------- Samples Plot and Preview ----------------------------------

#* generate
if SAMPLE_STRATEGY == "uniform":
    pairs = gen_abs_uniform(NUM_SAMPLES_LINE)
elif SAMPLE_STRATEGY == "uniform2":
    train_ds_raw = gen_abs_uniform(NUM_SAMPLES_LINE_1)
    test_ds = gen_abs_uniform(NUM_SAMPLES_LINE_2)
    train_ds = remove_duplicates_in_first(train_ds_raw, test_ds)
elif SAMPLE_STRATEGY == "randEachArea":
    train_pairs, test_pairs = genRandUniformABSamples(NUM_EACH_AREA_TRAIN), genRandUniformABSamples(NUM_EACH_AREA_TEST)
    pairs = np.concatenate((train_pairs, test_pairs), axis=0)

#* plot
if SAMPLE_STRATEGY == "uniform" or SAMPLE_STRATEGY == "randEachArea":
    x, y = zip(*pairs)
    plt.scatter(x, y)
elif SAMPLE_STRATEGY == "uniform2":
    x1, y1 = zip(*train_ds)
    plt.scatter(x1, y1, color="k")
    x2, y2 = zip(*test_ds)
    plt.scatter(x2, y2, color="g")
    train_ds.extend(test_ds)
    pairs = train_ds

for i, (xi, yi) in enumerate(pairs):
    plt.annotate(str(i), (xi, yi))

x = np.linspace(0, 1, 50)
y = 1/3*x + 1/3
plt.plot(x, y, color="r")
y = 3*x - 1
plt.plot(x, y, color="b")
plt.ylim([-0.1, 1.1])

#* save and show
plt.savefig(ds_savePathPre+".png")
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
for a_val, b_val in pairs:
    dmx = np.array(dm_convex.subs([(a, a_val), (b, b_val)]))
    dmx_dm = DensityMatrix(dmx)
    label = 0 if is_entangled(Matrix(dmx_dm.data), [2, 2]) else 1 # entangled as 0, separable as 1
    ds.append((dmx, label))

#! ---------------------------------- Save Data ----------------------------------
#* save
np.save(ds_savePathPre+".npy", ds)
np.save(ds_savePathPre+"_abPairs.npy", pairs)

#* notification
print("Data generation completed and saved.")