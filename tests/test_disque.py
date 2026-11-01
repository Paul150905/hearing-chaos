import time
import numpy as np
import scipy.sparse.linalg as sla
from scipy.special import jn_zeros
import matplotlib.pyplot as plt
from src.billiards.domains import masque_quart_stade
from src.billiards.grid import laplacian_domain

r = 1.0          # rayon du disque
E_MAX = 3000     # même plage d'énergies que pour le stade
espacement = 4 * np.pi / (np.pi * r**2 / 4)   # écart moyen entre niveaux (loi de Weyl)

# Énergies exactes du quart de disque (proposition 12) : m pair >= 2
exactes = np.concatenate([jn_zeros(m, 40)**2 / r**2 for m in range(2, 120, 2)])
exactes = np.sort(exactes[exactes < E_MAX])
K = len(exactes)
print(f"{K} niveaux exacts sous E = {E_MAX} ; écart moyen = {espacement:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for h in [0.02, 0.01, 0.005]:
    masque, x, y = masque_quart_stade(0.0, r, h)          # a = 0 : quart de disque
    L = laplacian_domain(masque, h)
    debut = time.time()
    calcul = np.sort(sla.eigsh(L, k=K, sigma=0, which="LM", return_eigenvectors=False))
    duree = time.time() - debut

    ecart = calcul - exactes                              # erreur sur chaque niveau
    relative = ecart / exactes
    # Partie régulière : tendance lissée de l'erreur relative (polynôme de degré 2 en E)
    tendance = np.polyval(np.polyfit(exactes, relative, 2), exactes)
    fluctuation = (relative - tendance) * exactes         # partie variable, en énergie

    print(f"--- h = {h} ({masque.sum()} points, eigsh en {duree:.1f} s)")
    print(f"  erreur relative moyenne : {100 * relative.mean():.3f} %")
    print(f"  partie variable / écart moyen : écart-type {fluctuation.std() / espacement:.3f}, "
          f"max {np.abs(fluctuation).max() / espacement:.3f}")

    ax[0].plot(exactes, 100 * relative, ".", markersize=3, label=f"h = {h}")
    ax[1].plot(exactes, fluctuation / espacement, ".", markersize=3, label=f"h = {h}")

ax[0].set_xlabel("Énergie exacte"); ax[0].set_ylabel("Erreur relative (%)")
ax[0].set_title("Erreur totale"); ax[0].legend()
ax[1].set_xlabel("Énergie exacte"); ax[1].set_ylabel("Partie variable / écart moyen")
ax[1].set_title("Partie variable de l'erreur"); ax[1].legend()
fig.tight_layout()
fig.savefig("figures/disque_erreur.png", dpi=150)
print("Figure enregistrée dans figures/disque_erreur.png")