import numpy as np

def npMxSimplify(matrix, tol=1e-12, num_decimal=2):
    # handle real and imaginary parts separately for complex arrays
    if np.iscomplexobj(matrix):
        real_part = np.where(np.abs(matrix.real) < tol, 0, matrix.real)
        imag_part = np.where(np.abs(matrix.imag) < tol, 0, matrix.imag)
        matrix = real_part + 1j * imag_part
    else:
        matrix = np.where(np.abs(matrix) < tol, 0, matrix)

    # round to the desired num_decimal
    matrix = np.round(matrix, num_decimal)

    return matrix

def spMxSimplify(matrix, tol=1e-12, num_decimal=2):
    """Simplify the matrix by removing the small values"""
    pass
