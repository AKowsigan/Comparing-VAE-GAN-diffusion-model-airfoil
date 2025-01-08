import numpy as np
import matplotlib.pyplot as plt

# Load the losses from the loss.npz file
data = np.load("gan/results/loss.npz")
D_losses = data['arr_0']  # Discriminator losses
G_losses = data['arr_1']  # Generator losses

# Create the loss plot
plt.figure(figsize=(10, 5))
plt.title("Loss Evolution During Training")
plt.plot(D_losses, label="Discriminator Losses (D)")
plt.plot(G_losses, label="Generator Losses (G)")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.grid()

# Save the plot before displaying it
output_file = "gan/images/loss.png"
plt.savefig(output_file, bbox_inches="tight")
print(f"Plot saved to: {output_file}")

# Display the plot
plt.show()
