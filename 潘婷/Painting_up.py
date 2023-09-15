#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def func(x, y):
    return 0.3* (np.cos(x))**2 * (np.cos(y))**2 \
    + np.cos(2*x) + np.cos(2*y)
    

def down_surface(x,y):
    return 0 * x - 1

plt.style.use('_mpl-gallery')

x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
x, y = np.meshgrid(x, y)
z1 = func(x, y)
# z2 = down_surface(x,y)
#%%



#%%


# Plot the surface
fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z1, cmap=cm.Blues)
# ax.plot_surface(x, y, z2, cmap=cm.Blues)

# ax.set(xticklabels=[],
#        yticklabels=[],
#        zticklabels=[])
ax.axis('off')
plt.show()
# plt.plot(x, y)
# plt.show()
# %%
