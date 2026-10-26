import numpy as np
import scipy.sparse.linalg as sla
from src.billiards.grid import laplacian_1d

# La grille de la partie 2.1 : n points intérieurs, h = L / (n + 1)
n, L = 200, 1.0
h = L / (n + 1)
D = laplacian_1d(n, h)

# Les 10 plus petites valeurs propres de D, triées
valeurs = np.sort(sla.eigsh(D, k=10, sigma=0, which="LM", return_eigenvectors=False))

# Les deux références de l'article : formule (34) et énergies (20)
m = np.arange(1, 11)
discretes = (4 / h**2) * np.sin(m * np.pi * h / (2 * L))**2
continues = (m * np.pi / L)**2

print("Test du code (~1e-14) :", np.max(np.abs(valeurs - discretes) / discretes))
print("Erreur de grille en % :", np.round(100 * (continues - valeurs) / continues, 4))
print("Prédiction (kh)²/12 en % :", np.round(100 * (m * np.pi * h)**2 / 12, 4))