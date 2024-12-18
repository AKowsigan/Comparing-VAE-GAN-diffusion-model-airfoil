import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from vae.model import VAE

# Create a directory to save results
os.makedirs("vae/results", exist_ok=True)

# Load the dataset
coords_npz = np.load("dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]

# Flatten the data for the VAE input
input_dim = coords.shape[1] * coords.shape[2]
coords = coords.reshape(coords.shape[0], -1)
dataset = TensorDataset(torch.tensor(coords, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the VAE model
latent_dim = 3
vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Loss function for VAE
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (mean squared error)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # Kullback-Leibler divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# Training parameters
epochs = 50000
save_interval = 5000
vae.train()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        # Forward pass through the VAE
        recon_x, mu, logvar = vae(x)
        # Compute the loss
        loss = vae_loss(recon_x, x, mu, logvar)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Display training progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset):.4f}")

    # Save the model every 5000 epochs
    if (epoch + 1) % save_interval == 0:
        model_path = f"vae/results/vae_model_{epoch + 1}.pth"
        torch.save(vae.state_dict(), model_path)
        print(f"Model saved to {model_path}")

# Save the final model
final_model_path = "vae/results/vae_model_final.pth"
torch.save(vae.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")
