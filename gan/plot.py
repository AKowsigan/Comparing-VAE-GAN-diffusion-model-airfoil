import numpy as np
import matplotlib.pyplot as plt

# Charger les pertes depuis le fichier loss.npz
data = np.load("gan/results/loss.npz")
D_losses = data['arr_0']  # Les pertes du discriminant
G_losses = data['arr_1']  # Les pertes du générateur

# Créer le graphique des pertes
plt.figure(figsize=(10, 5))
plt.title("Évolution des pertes pendant l'entraînement")
plt.plot(D_losses, label="Pertes du Discriminant (D)")
plt.plot(G_losses, label="Pertes du Générateur (G)")
plt.xlabel("Épochs")
plt.ylabel("Pertes")
plt.legend()
plt.grid()
plt.show()
plt.savefig("gan/results/loss.png")
