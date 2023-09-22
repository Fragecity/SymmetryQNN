#%% importing
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as qnp
import numpy as np
from numba import vectorize, types, njit
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
#%%

dev = qml.device('default.mixed', wires = 1)
RHO = np.array([[1, 0], [0, 0]])
Z = np.array([[1, 0], [0, -1]])


# @vectorize
# @njit(parallel=True)
@qml.qnode(device=dev)
def pqc(theta, phi):
    qml.RX(theta, wires=0)
    qml.RY(phi, wires=0)
    
    O = qml.Hermitian(RHO, wires=0)
    return qml.expval(O)
    
def sg(theta, phi):
    U = qml.matrix(pqc)(theta, phi)
    O_tilde = U.conj().T @ RHO @ U
    twiled_O = (O_tilde + Z @ O_tilde @ Z)/2
    diff = O_tilde - twiled_O
    res = np.trace(diff @ diff)
    return res.real

# %% set_data
# @njit(parallel=True)
def run():
    grid_size = 64

    theta = np.linspace(-np.pi, np.pi, grid_size)
    phi = np.linspace(-np.pi, np.pi, grid_size)

    theta, phi = np.meshgrid(theta, phi)
    cost_value = np.zeros_like(theta)
    gd = np.zeros_like(theta)
    cost_value = np.zeros_like(theta)
    gd = np.zeros_like(theta)
    for i, j in product(list(range(grid_size)), list(range(grid_size))):
        
        cost_value[i, j] = 1 - pqc(theta[i, j], phi[i, j])
        
        gd[i, j] = sg(theta[i, j], phi[i, j])  # To Edit
        
    plot_lst = np.zeros_like(theta)
    for i, j in product(list(range(grid_size)), list(range(grid_size))):
        temp = cost_value[i, j] + gd[i,j]
        if temp < 0.75:
            plot_lst[i, j] = temp
        else:
            plot_lst[i, j] = np.NaN
# strategy1 = cost_value + 
    
#%% plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    x = 1.3
    y = -1.7
    eps = 0.05
    rate = 12
    ax.scatter([x, x], [y, y], [-1.3+eps, pqc(-1,0.5)+eps], marker='o', color='red')
    dx = (pqc(x+eps, y) - pqc(x-eps, y)) / (2*eps)
    dy = (pqc(x, y+eps) - pqc(x, y-eps)) / (2*eps)
    # 绘制 3D 矢量
    ax.quiver([x], [y], [-1.3], [dx], [dy], [0], color='red', length=1, arrow_length_ratio=0.3, normalize=False)
    
    
    
    surf = ax.plot_surface(theta, phi, cost_value, cmap='viridis')
    ax.contourf(theta, phi, cost_value, offset=-1.3, cmap='viridis', levels=20)
    ax.contourf(theta, phi, plot_lst, offset=-1.3, cmap='Wistia', levels=20, alpha=0.6)

    ax.set_zlim(-1.3, 1)

    ax.grid(False)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    # ax.set_zlabel('cost', position=(0.5, 0.5))
    fig.colorbar(surf, shrink=0.5, aspect=8)
    

#%% 绘制点和箭头

    

    # print(dx, dy)


# %% 设置坐标轴
    xticks = [-np.pi, 0, np.pi]
    xticks_label = [r'$-\pi$', r'$0$', r'$\pi$']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_label)
    yticks = [-np.pi, 0, np.pi]
    yticks_label = [r'$-\pi$', r'$0$', r'$\pi$']
    ax.set_yticks(yticks)  
    ax.set_yticklabels(yticks_label)
    
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.set_box_aspect([1,1,0.6])
    plt.show()
# %%
if __name__ == '__main__':

    run()