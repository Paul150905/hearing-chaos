import numpy as np


def masque_quart_stade(a, r, h):
    """Masque du quart de stade Q_{a,r} sur le rectangle [0, a+r] x [0, r], de pas h.
    h doit diviser r et a + r. Renvoie le masque et les coordonnées x, y."""
    nx = round((a + r) / h) - 1                       # points intérieurs en x
    ny = round(r / h) - 1                             # points intérieurs en y
    x = h * np.arange(1, nx + 1)
    y = h * np.arange(1, ny + 1)
    X, Y = np.meshgrid(x, y)                          # tableaux de taille (ny, nx)
    masque = np.maximum(X - a, 0)**2 + Y**2 < r**2    # inégalité (1)
    return masque, x, y

