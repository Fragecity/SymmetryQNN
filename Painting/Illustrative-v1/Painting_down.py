import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')
def func(x, y):
    return 0.3* (np.cos(x))**2 * (np.cos(y))**2 \
    + np.cos(2*x) + np.cos(2*y)

def symmetry_area(x,y):
    return (x-np.pi/2)**2 + (y-np.pi/2)**2
# make data

    

x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
x,y = np.meshgrid(x,y)
z = func(x,y)

# s = symmetry_area(x,y)
levels = np.linspace(z.min(), z.max(), 17)
# levels_sym = np.linspace(s.min(), s.max(), 11)

# plot
fig, ax = plt.subplots()

ax.contourf(x,y,z, levels=levels)
# ax.contourf(x,y,s, levels=levels_sym, cmap='Reds')
ax.axis('off')
plt.show()