import time
import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from src.billiards.domains import masque_quart_stade
from src.billiards.grid import laplacian_domain

a, r, h = 1.0, 1.0, 0.005   # quart de stade, pas validé sur le disque
K = 400                     # nombre de niveaux voulus

# 1. La matrice
masque, x, y = masque_quart_stade(a, r, h)
L = laplacian_domain(masque, h)
print(f"Quart de stade a = {a}, r = {r}, h = {h} : N = {masque.sum()} points")

# 2. Les K plus petites énergies, avec leurs ondes
debut = time.time()
E, V = sla.eigsh(L, k=K, sigma=0, which="LM")
ordre = np.argsort(E)
E, V = E[ordre], V[:, ordre]
print(f"{K} niveaux calculés en {time.time() - debut:.0f} s ; E max = {E[-1]:.1f}")

# 3. Enregistrer les énergies (le dossier data/ n'est pas envoyé sur GitHub)
np.savez(f"data/stade_a{a}_r{r}_h{h}.npz", E=E, a=a, r=r, h=h)

# 4. Contrôle de complétude : nombre de niveaux sous E, comparé à la loi de Weyl
A = a * r + np.pi * r**2 / 4                 # aire du quart de stade
P = a + np.pi * r / 2 + (a + r) + r          # périmètre : segment, arc, deux axes
n = np.arange(1, K + 1)                      # N(E_n) = n
weyl = A * E / (4 * np.pi) - P * np.sqrt(E) / (4 * np.pi)
fluct = n - weyl
print(f"N(E) - Weyl : moyenne {fluct.mean():.2f}, min {fluct.min():.2f}, max {fluct.max():.2f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(E, fluct, lw=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Énergie $E$"); ax.set_ylabel("$N(E)$ − Weyl")
ax.set_title("Quart de stade : contrôle de complétude")
fig.tight_layout(); fig.savefig("figures/stade_weyl.png", dpi=150)

# 5. Quelques ondes, dépliées sur le stade entier par symétrie
def stade_entier(v):
    q = np.zeros(masque.shape)
    q[masque] = v                            # remise en forme du vecteur (reshape)
    haut = np.hstack([-q[:, ::-1], q])       # impaire en x
    return np.vstack([-haut[::-1, :], haut]) # impaire en y

niveaux = [1, 2, 10, 50, 200, 400]
fig, axes = plt.subplots(2, 3, figsize=(12, 4.5))
for axe, m in zip(axes.ravel(), niveaux):
    phi = stade_entier(V[:, m - 1])
    axe.imshow(phi**2, origin="lower", cmap="inferno",
               extent=[-(a + r), a + r, -r, r])
    axe.set_title(f"niveau {m}, E = {E[m - 1]:.0f}", fontsize=9)
    axe.set_xticks([]); axe.set_yticks([])
fig.tight_layout(); fig.savefig("figures/stade_ondes.png", dpi=150)
print("Figures enregistrées : figures/stade_weyl.png et figures/stade_ondes.png")