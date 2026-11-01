import numpy as np
import scipy.sparse as sp


def laplacian_1d(n, h):
    """Matrice (n x n) de -d²/dx², psi = 0 aux deux bords."""
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


def laplacian_domain(masque, h):
    """Matrice de -laplacien sur un domaine quelconque, psi = 0 hors du masque.
    masque[j, i] vaut True si le point (x_i, y_j) est dans le domaine."""
    ny, nx = masque.shape
    LR = laplacian_rectangle(nx, ny, h, h).tocsr()   # rectangle englobant
    garde = masque.ravel()                            # points du masque, ligne par ligne
    return LR[garde][:, garde].tocsc()                # proposition 9