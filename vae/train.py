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
from vae.model import VAE
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

# Load dataset
perfs_npz = np.load("dataset/standardized_perfs.npz")
coords_npz = np.load("dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]

dataset = TensorDataset(torch.tensor(coords))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Initialize VAE
vae = VAE(input_dim=opt.coord_size, latent_dim=opt.latent_dim)
if cuda:
    print("Using GPU")
    vae.cuda()

# Optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
start = time.time()
losses = []
for epoch in range(opt.n_epochs):
    epoch_loss = 0
    for i, (coords,) in enumerate(dataloader):
        real_coords = Variable(coords.type(FloatTensor))

        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(real_coords)

        loss = vae_loss(reconstructed, real_coords, mu, logvar)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"[Epoch {epoch + 1}/{opt.n_epochs}] [Loss: {avg_loss:.6f}]")

    if (epoch + 1) % 1000 == 0:
        torch.save(vae.state_dict(), f"vae/results/vae_params_{epoch + 1}.pth")

torch.save(vae.state_dict(), f"vae/results/vae_params_{opt.n_epochs}.pth")
np.savez("vae/results/losses.npz", np.array(losses))

end = time.time()
print(f"Training Time: {(end - start) / 60:.2f} minutes")

plt.plot(range(len(losses)), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("vae/results/training_loss_curve.png")
plt.show()
