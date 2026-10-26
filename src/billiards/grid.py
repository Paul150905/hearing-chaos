import numpy as np
import scipy.sparse as sp


def laplacian_1d(n, h):
    diagonale = 2 * np.ones(n)
    voisine = -np.ones(n - 1)
    D = sp.diags([voisine, diagonale, voisine], [-1, 0, 1])
    return (D / h**2).tocsc()
