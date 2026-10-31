import numpy as np
import scipy.sparse as sp


def laplacian_1d(n, h):
    diagonale = 2 * np.ones(n)
    voisine = -np.ones(n - 1)
    D = sp.diags([voisine, diagonale, voisine], [-1, 0, 1])
    return (D / h**2).tocsc()

def laplacian_rectangle(nx, ny, hx, hy):
    """Matrice (nx*ny x nx*ny) de -laplacien sur un rectangle, psi = 0 au bord.
    Points numérotés ligne par ligne."""
    Dx = laplacian_1d(nx, hx)          # 1D le long d'une ligne
    Dy = laplacian_1d(ny, hy)          # 1D le long d'une colonne
    Lx = sp.kron(sp.identity(ny), Dx)  # une copie de Dx par ligne
    Ly = sp.kron(Dy, sp.identity(nx))  # Dy, nombres -> blocs identité
    return (Lx + Ly).tocsc()

