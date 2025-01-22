if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from vae.model import VAE

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class EvalVAE:
    def __init__(self, MODEL_PATH, coords_npz):
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        self.model = VAE(input_dim=496, latent_dim=16)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if cuda:
            self.model.cuda()

        self.coords = {
            'data': coords_npz[coords_npz.files[0]],
            'mean': coords_npz[coords_npz.files[1]],
            'std': coords_npz[coords_npz.files[2]],
        }

    def rev_standardize(self, coords):
        return coords * self.coords['std'] + self.coords['mean']

    def reconstruct(self, data):
        real_coords = Variable(torch.tensor(data).type(FloatTensor))
        reconstructed, _, _ = self.model(real_coords)
        return reconstructed.cpu().detach().numpy()

if __name__ == "__main__":
    coords_npz = np.load("dataset/standardized_coords.npz")
    MODEL_PATH = "vae/results/vae_params_10000.pth"
    eval_vae = EvalVAE(MODEL_PATH, coords_npz)

    sample_coords = eval_vae.coords['data'][:12]
    reconstructed_coords = eval_vae.reconstruct(sample_coords)
    sample_coords = eval_vae.rev_standardize(sample_coords)
    reconstructed_coords = eval_vae.rev_standardize(reconstructed_coords)

    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    fig.suptitle("VAE: Original vs Reconstructed Profiles", fontsize=16)

    for i in range(12):
        x_orig, y_orig = sample_coords[i].reshape(2, -1)
        x_recon, y_recon = reconstructed_coords[i].reshape(2, -1)

        ax = axes[i // 3, i % 3]
        ax.plot(x_orig, y_orig, label="Original", linestyle='--')
        ax.plot(x_recon, y_recon, label="Reconstructed", linestyle='-')
        ax.set_title(f"Profile {i+1}", fontsize=10)
        ax.axis('equal')
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = "vae/results/reconstructed_profiles.png"
    plt.savefig(output_file)
    plt.show()
