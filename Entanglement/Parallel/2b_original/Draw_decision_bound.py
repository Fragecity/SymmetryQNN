import matplotlib.pyplot as plt
from matplotlib import cm
import pennylane as qml
import numpy as np
import pickle
from DataGener import SysSwapSymmetry, werner, dagger
from Train_2b import cost, symmetry, circuit
from itertools import product
with open('res 2b (0).pkl', 'rb') as f:
    [cst_lst, cstG_lst, para_lst, paraG_lst] = pickle.load(f)


from matplotlib.animation import FuncAnimation
# from Tools import np, plt, trace
# from Werner import werner, witness, QNN_for_Werner
# import json
# with open('Config.json') as json_file:
#     CONFIG = json.load(json_file)
# fig,ax = plt.subplot()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x_lst = np.linspace(-1,1, 32)
y_ref = np.zeros_like(x_lst)


# U = qml.matrix(circuit)([X[i,j], Y[i,j]])

# for x in x_lst:
#     rho = werner(1, x)

# def get_expects(O):
#     return [ trace(
#     werner(CONFIG['num_part'], f) @ O).real for f in x_lst]

# y_ref = get_expects(W)

# line1, = ax.plot(x_lst, y_ref)
line2, = ax.plot(x_lst, y_ref)
line3, = ax.plot(x_lst, y_ref)
ax.set_ylim(-1.1, 1.1)

# 清空当前帧
# def init():
#     line.set_ydata([np.nan] * len(x_lst))
#     return line,

# 更新新一帧的数据
def update(t):
    line2.set_ydata(get_y(para_lst[t]))
    line3.set_ydata(get_y(paraG_lst[t]))
    return line2, line3

def get_y(parameter):
    U = qml.matrix(circuit)(parameter)
    return [np.trace(U @ werner(1, f) @ dagger(U) @ symmetry.observable) for f in x_lst]


# 调用 FuncAnimation
ani = FuncAnimation(fig
                   ,update
                #    ,init_func=init
                   ,frames=64
                   ,interval=4
                   ,blit=True
                   )

ani.save("animation2.gif", fps=6)
plt.plot()