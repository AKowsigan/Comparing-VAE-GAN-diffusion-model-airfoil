if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import torch.nn as nn
import torch
from ae.models import AutoEncoder
from util import save_loss, to_cuda

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--coord_size", type=int, default=496, help="size of input coordinate dimension")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
opt = parser.parse_args()

coord_shape = (1, opt.coord_size)
cuda = True if torch.cuda.is_available() else False

# Configure data loader
perfs_npz = np.load("dataset/standardized_perfs.npz")
coords_npz = np.load("dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]
perfs = perfs_npz[perfs_npz.files[0]]

dataset = TensorDataset(torch.tensor(coords))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Initialize AutoEncoder
autoencoder = AutoEncoder()
if cuda:
    print("Using GPU")
    autoencoder.cuda()

# Optimizer and loss function
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
criterion = nn.MSELoss()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Training loop
start = time.time()
losses = []
for epoch in range(opt.n_epochs):
    epoch_loss = 0
    for i, (coords,) in enumerate(dataloader):
        # Configure input
        real_coords = Variable(coords.type(FloatTensor))

        # -----------------
        #  Train AutoEncoder
        # -----------------
        optimizer.zero_grad()

        # Forward pass
        reconstructed_coords, latent_vectors = autoencoder(real_coords)

        # Compute reconstruction loss
        loss = criterion(reconstructed_coords, real_coords)
        epoch_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    # Print training progress
    print(f"[Epoch {epoch + 1}/{opt.n_epochs}] [Reconstruction Loss: {avg_loss:.6f}]")

    # Save model checkpoints periodically
    if (epoch + 1) % 1000 == 0:
        torch.save(autoencoder.state_dict(), f"ae/results/autoencoder_params_{epoch + 1}.pth")

# Save final model
torch.save(autoencoder.state_dict(), f"ae/results/autoencoder_params_{opt.n_epochs}.pth")

# Save training losses
np.savez("ae/results/losses.npz", np.array(losses))

end = time.time()
print(f"Training Time: {(end - start) / 60:.2f} minutes")

# Plot training loss curve
plt.plot(range(len(losses)), losses, label="Reconstruction Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("ae/results/training_loss_curve.png")
plt.show()
