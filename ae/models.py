
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.nn as nn
import torch

coord_shape = (1, 496)  # 248 x 2 for x and y
latent_dim = 16  # Adjustable latent space dimension


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(496, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, latent_dim),  # Map to latent_dim
        )

    def forward(self, coords):
        # Flatten the input coordinates
        coords_flat = coords.view(coords.size(0), -1)
        z = self.model(coords_flat)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 496),  # Map back to original dimensions
            nn.Tanh(),  # Keep output normalized
        )

    def forward(self, z):
        coords_flat = self.model(z)
        coords = coords_flat.view(coords_flat.size(0), *coord_shape)
        return coords


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, coords):
        z = self.encoder(coords)  # Encode to latent space
        reconstructed_coords = self.decoder(z)  # Decode back to original shape
        return reconstructed_coords, z
