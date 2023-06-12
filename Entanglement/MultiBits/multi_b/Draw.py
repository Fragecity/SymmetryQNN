#%%
import pickle
with open('./data/result cost 2023-03-21 10-21.pkl', 'rb') as f:
    [x, y] = pickle.load(f)
with open('./data/result costG 2023-03-21 10-27.pkl', 'rb') as f:
    [xG, yG] = pickle.load(f)


#%%
from matplotlib.animation import FuncAnimation
from Tools import np, plt, trace
from Werner import werner, witness, QNN_for_Werner
import json
with open('Config.json') as json_file:
    CONFIG = json.load(json_file)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x_lst = np.linspace(-1,1, 128)

qnn = QNN_for_Werner(CONFIG['num_part'], CONFIG['num_layer'], 0)
W = witness(CONFIG['num_part'])

def get_expects(O):
    return [ trace(
    werner(CONFIG['num_part'], f) @ O).real for f in x_lst]

y_ref = get_expects(W)

line1, = ax.plot(x_lst, y_ref)
line2, = ax.plot(x_lst, y_ref)
line3, = ax.plot(x_lst, y_ref)
ax.set_ylim(-1.1, 1.1)

# 清空当前帧
# def init():
#     line.set_ydata([np.nan] * len(x_lst))
#     return line,

# 更新新一帧的数据
def update(t):
    O2 = qnn.get_O_tilde(x[t])
    O3 = qnn.get_O_tilde(xG[t])
    line2.set_ydata(get_expects(O2))
    line3.set_ydata(get_expects(O3))
    return line1, line2, line3

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