if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from ae.models import AutoEncoder
from sklearn.metrics import mean_squared_error

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Eval:
    def __init__(self, MODEL_PATH, coords_npz):
        # Load the AutoEncoder model
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        self.model = AutoEncoder()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if cuda:
            self.model.cuda()
        
        # Load dataset normalization parameters
        self.coords = {
            'data': coords_npz[coords_npz.files[0]],
            'mean': coords_npz[coords_npz.files[1]],
            'std': coords_npz[coords_npz.files[2]],
        }

    def rev_standardize(self, coords):
        """Reverse standardize the coordinates."""
        return coords * self.coords['std'] + self.coords['mean']

    def reconstruct_coords(self, data):
        """Reconstruct coordinates using the AutoEncoder."""
        real_coords = Variable(torch.tensor(data).type(FloatTensor))
        reconstructed, _ = self.model(real_coords)
        return reconstructed.cpu().detach().numpy()

    def calculate_mse(self, original, reconstructed):
        """Calculate MSE for original and reconstructed profiles."""
        mse_values = []
        for orig, recon in zip(original, reconstructed):
            mse = mean_squared_error(orig, recon)
            mse_values.append(mse)
        avg_mse = np.mean(mse_values)
        return mse_values, avg_mse


if __name__ == "__main__":
    # Load dataset
    coords_npz = np.load("dataset/standardized_coords.npz")
    MODEL_PATH = "ae/results/autoencoder_params_10000.pth"
    evl = Eval(MODEL_PATH, coords_npz)

    # Use a sample from the dataset
    sample_coords = evl.coords['data'][:12]
    reconstructed_coords = evl.reconstruct_coords(sample_coords)

    # Reverse standardize
    sample_coords = evl.rev_standardize(sample_coords)
    reconstructed_coords = evl.rev_standardize(reconstructed_coords)

    # Calculate MSE
    mse_values, avg_mse = evl.calculate_mse(sample_coords, reconstructed_coords)
    print("Average MSE across profiles: {:.6f}".format(avg_mse))


    # Plot original vs reconstructed profiles
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    fig.suptitle("AutoEncoder: Original vs Reconstructed Profiles", fontsize=16)

    for i in range(12):
        x_orig, y_orig = sample_coords[i].reshape(2, -1)
        x_recon, y_recon = reconstructed_coords[i].reshape(2, -1)

        ax = axes[i // 3, i % 3]
        ax.plot(x_orig, y_orig, label="Original", linestyle='--')
        ax.plot(x_recon, y_recon, label="Reconstructed", linestyle='-')
        ax.set_title("Profile {}\nMSE: {:.6f}".format(i + 1, mse_values[i]), fontsize=10)
        ax.axis('equal')
        ax.grid(True)
        ax.legend(fontsize=8)

    # Save the visualization
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = "ae/results/reconstructed_profiles_with_mse.png"
    plt.savefig(output_file)
    print("Saved reconstructed profiles with MSE to {}".format(output_file))
    plt.show()
