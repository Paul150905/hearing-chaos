import numpy as np
from src.billiards.grid import laplacian_rectangle, laplacian_domain

# 1. Un masque entièrement vrai doit redonner le rectangle
masque = np.ones((3, 4), dtype=bool)
ecart = abs(laplacian_domain(masque, 1.0) - laplacian_rectangle(4, 3, 1.0, 1.0)).max()
print("Masque plein = rectangle :", ecart == 0)

# 2. La grille en L de l'article (point 6 retiré)
masque = np.array([[True, True, True],      # ligne du bas : points 1, 2, 3
                   [True, True, False]])    # ligne du haut : points 4, 5, (6 retiré)
L = laplacian_domain(masque, 1.0).toarray()
attendu = np.array([[ 4, -1,  0, -1,  0],
                    [-1,  4, -1,  0, -1],
                    [ 0, -1,  4,  0,  0],
                    [-1,  0,  0,  4, -1],
                    [ 0, -1,  0, -1,  4]])
print("Grille en L conforme :", np.array_equal(L, attendu))
print(L)