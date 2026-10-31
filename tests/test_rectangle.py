import time
import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from src.billiards.grid import laplacian_rectangle

# Le rectangle de validation : a = 1, b = pi/3 (pas de dégénérescence)
a, b = 1.0, np.pi / 3
K = 200                                   # nombre de niveaux comparés


def spectres_exacts(nx, ny):
    """Pour une grille nx x ny : les K plus petites énergies discrètes (formule 41)
    et, pour les mêmes couples (m, n), les énergies continues (formule 42) et k*h."""
    hx, hy = a / (nx + 1), b / (ny + 1)
    m = np.arange(1, nx + 1)[:, None]     # m en colonne
    n = np.arange(1, ny + 1)[None, :]     # n en ligne : tableau de tous les couples (m, n)
    discret = (4 / hx**2) * np.sin(m * np.pi * hx / (2 * a))**2 \
            + (4 / hy**2) * np.sin(n * np.pi * hy / (2 * b))**2
    continu = np.pi**2 * (m**2 / a**2 + n**2 / b**2)
    kh = np.maximum(m * np.pi / a * hx, n * np.pi / b * hy)   # le pire des deux k*h
    ordre = np.argsort(discret, axis=None)[:K]                 # les K plus petits niveaux
    return discret.ravel()[ordre], continu.ravel()[ordre], kh.ravel()[ordre]


for nx, ny in [(300, 300), (100, 100)]:
    hx, hy = a / (nx + 1), b / (ny + 1)
    L = laplacian_rectangle(nx, ny, hx, hy)

    debut = time.time()
    calcul = np.sort(sla.eigsh(L, k=K, sigma=0, which="LM", return_eigenvectors=False))
    duree = time.time() - debut

    discret, continu, kh = spectres_exacts(nx, ny)
    ecart_code = np.max(np.abs(calcul - discret) / discret)
    erreur = 100 * (continu - discret) / continu          # erreur de grille, en %

    print(f"--- Grille {nx} x {ny} ({nx * ny} points, eigsh en {duree:.1f} s)")
    print(f"Test du code (~1e-12) : {ecart_code:.1e}")
    print(f"Erreur de grille max sur {K} niveaux : {erreur.max():.3f} %")
    depasse = np.where(erreur > 1)[0]
    if len(depasse):
        i = depasse[0]
        print(f"1 % dépassé au niveau {i + 1}, pour k*h = {kh[i]:.3f}")
    else:
        print("1 % jamais dépassé")

    plt.plot(continu, erreur, ".", markersize=4, label=f"grille {nx} x {ny}")

plt.axhline(1, color="k", linestyle="--", linewidth=0.8, label="1 %")
plt.xlabel("Énergie continue $E_{m,n}$")
plt.ylabel("Erreur relative (%)")
plt.title("Rectangle $a = 1$, $b = \\pi/3$ : erreur de la grille")
plt.legend()
plt.savefig("figures/rectangle_erreur.png", dpi=150, bbox_inches="tight")
print("Figure enregistrée dans figures/rectangle_erreur.png")