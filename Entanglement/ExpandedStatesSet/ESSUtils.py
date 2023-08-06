import numpy as np
import sympy as sp
import pennylane as qml
import pennylane.numpy as qnp
from sympy import Matrix, symbols, MutableDenseMatrix
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sympy.physics.quantum import TensorProduct
from Utils import dagger, dmx_zero, bellState

#! --------------------------- Convex Combination Form --------------------------- !#
#* constants
phi_plus = Matrix(bellState("phi_plus"))
phi_minus = Matrix(bellState("phi_minus"))

#* matrix form
a, b = symbols('a b')
dm_convex = (1-a-b) * sp.eye(4)/4 + a * phi_minus * phi_minus.H + b * phi_plus * phi_plus.H

#! --------------------------- Samples Generation --------------------------- !#
# generate samples of a,b that satisfying 0=<a,b<=1 and 0<=a+b<=1
def gen_abs_uniform(num_samples):
    # generate a linear space of numbers between 0 and 1
    lin_space = np.linspace(0, 1, num_samples)

    # initialize list to store valid pairs
    valid_pairs = []

    # check each pair of numbers
    for i in range(num_samples):
        for j in range(num_samples):
            # if the sum is less or equal than 1, add the pair to the list
            if lin_space[i] + lin_space[j] <= 1:
                valid_pairs.append((lin_space[i], lin_space[j]))

    return valid_pairs

def remove_duplicates_in_first(parisA, pairsB):
    """remove shared paris of both set in parisA

    Returns:
        list: parisA - (parisA & pairsB)
    """
    set1 = set(parisA)
    set2 = set(pairsB)
    return list(set1 - (set1 & set2))

def fusePairs(pairsA, pairsB):
    """fuse pairs from two sets
    """
    set1 = set(pairsA)
    set2 = set(pairsB)
    return list(set1 | set2)

def sample_points_in_polygon(vertices, num_samples):
    """sample points in a convex polygon

    Args:
        vertices (np.ndarray): vertices set of the polygon
        num_samples (int, optional): number of samples

    PAT: might not sample points on boundaries
    """

    # * Triangulate the polygon
    triangulation = Delaunay(vertices)
    triangles = vertices[triangulation.simplices]  # vertices of the triangles, shape = [num_triangles, 3, 2]

    # Calculate areas of triangles
    areas = 0.5 * np.abs(np.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :]))

    # Normalize the areas so that they sum to 1
    areas /= np.sum(areas)

    # * select triangles for each sample
    triangle_indices = np.random.choice(len(triangles), size=num_samples, p=areas)

    # * gen samples
    # For each chosen triangle idx, generate (s, t), which corresponds to a point within the triangle
    st = np.random.rand(num_samples, 2)  # range [0, 1)
    # for each sample, if s + t > 1, then replace it with (1 - s, 1 - t)
    mask = st.sum(axis=1) > 1
    st[mask] = 1 - st[mask]

    #  for each sample, point constructed from (1 - s - t) * v0 + s * v1 + t * v2,
    # where vs correspond to vertices of these randomly chosen triangles
    points = (1 - st.sum(axis=1, keepdims=True)) * triangles[triangle_indices, 0, :] + \
             st[:, 0, None] * triangles[triangle_indices, 1, :] + \
             st[:, 1, None] * triangles[triangle_indices, 2, :]

    return points

def genRandUniformABSamples(num_eachArea):
    """Generate random samples of a, b for the convex combination form set of Werner state"""

    etgl_area_above = np.array([[0, 1 / 3], [1 / 2, 1 / 2], [0, 1]])
    etgl_area_below = np.array([[1 / 3, 0], [1 / 2, 1 / 2], [1, 0]])
    separable_area = np.array([[0, 0], [0, 1 / 3], [1 / 3, 0], [1 / 2, 1 / 2]])

    ds1 = sample_points_in_polygon(etgl_area_above, num_eachArea)
    ds2 = sample_points_in_polygon(etgl_area_below, num_eachArea)
    ds3 = sample_points_in_polygon(separable_area, num_eachArea)

    return np.concatenate((ds1, ds2, ds3), axis=0)

def genABsRandom(num_samples):
    """Generate random samples of a, b for the convex combination form set of Werner state"""
    ds = sample_points_in_polygon(np.array([[0, 0], [0, 1], [1, 0]]), num_samples)
    return ds

#！ --------------------------- model --------------------------- !#
def sigmoid(x):
    return 1 / (1 + qnp.exp(-x))


#! --------------------------- Plot Generated Lines --------------------------- !#
def roundCoeffsOfSpExpr(LineExpr):
    """Return the coeffs rounder expression"""
    coeffs = LineExpr.as_coefficients_dict()
    map = {symbol: round(coeff, 2) for symbol, coeff in coeffs.items()}
    return sum([symbol * coeff for symbol, coeff in map.items()])

def plotGeneratedLines(dmx_sp, ansatz1, ansatz2, params1, params2, Ob1, Ob2, abPairs):

    U1 = qml.matrix(ansatz1)(params1)
    U2 = qml.matrix(ansatz2)(params2)

    if U1.shape == (16, 16):
        rho_zero = dmx_zero(2)
        dmx_sp = TensorProduct(dmx_sp, Matrix(rho_zero))

    # trace output
    tr1 = sp.trace(Matrix(U1) @ dmx_sp @ Matrix(dagger(U1)) @ Matrix(Ob1))
    tr2 = sp.trace(Matrix(U2) @ dmx_sp @ Matrix(dagger(U2)) @ Matrix(Ob2))

    # data load and plot generated lines
    y1 = sp.solve(tr1, b)
    y1 = y1[0] if y1 != [] else 0
    y2 = sp.solve(tr2, b)
    y2 = y2[0] if y2 != [] else 0

    def plotOriginal(ax):
        x = np.linspace(0, 1, 50)
        #* target two lines
        y = 1/3*x + 1/3
        ax.plot(x, y, color="r", label=f"$b = 1/3*a + 1/3$")
        y = 3*x - 1
        ax.plot(x, y, color="r", label=f"$b = 3*a - 1$")
        #* learned two lines
        if y1 != 0:
            y = [y1.subs(a, xi) for xi in x]
            ax.plot(x, y, color="b", label=f'$b_1={roundCoeffsOfSpExpr(y1)}$')
        else:
            ax.plot(x, [0]*len(x), color="b", label=f'$b_1=0$')
        if y2 != 0:
            y = [y2.subs(a, xi) for xi in x]
            ax.plot(x, y, color="g", label=f'$b_2={roundCoeffsOfSpExpr(y2)}$')
        else:
            ax.plot(x, [0]*len(x), color="g", label=f'$b_2=0$')
        #* samples
        x, y = zip(*abPairs)
        ax.scatter(x, y)
        # annotation
        # for i, (xi, yi) in enumerate(pairs):
            # plt.annotate(str(i), (xi, yi))

    fig, ax = plt.subplots(1,2, figsize=(12, 5))

    plotOriginal(ax[0])
    plotOriginal(ax[1])
    ax[1].set_ylim([-0.1, 1.1])
    ax[1].legend(loc="upper right")

    return fig, ax