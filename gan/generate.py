if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from torch.autograd import Variable
import torch
from gan.models import Generator
from gan.utils import to_cpu
import matplotlib.pyplot as plt

# Check if CUDA is available
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def load_generator(model_path, latent_dim=3):
    """Load the pre-trained generator model."""
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    generator = Generator(latent_dim)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

def generate_airfoil(generator, cl, latent_dim=3, data_num=1):
    """
    Generate airfoil shapes for a given target C_L.
    
    Args:
        generator (Generator): Pre-trained GAN generator.
        cl (float): Target lift coefficient (C_L).
        latent_dim (int): Dimensionality of the latent space.
        data_num (int): Number of airfoil shapes to generate.

    Returns:
        np.ndarray: Generated airfoil coordinates.
    """
    # Sample random noise vector (z)
    # z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, latent_dim))))
    z = Variable(FloatTensor(np.random.normal(0, 1, (1, latent_dim))))
    # Target C_L as labels
    # labels = Variable(FloatTensor(np.full((data_num, 1), cl)))
    labels = Variable(torch.reshape(FloatTensor([cl]), (1, 1)))
    # Generate airfoil coordinates
    gen_coords = to_cpu(generator(z, labels)).detach().numpy()
    #Print the shape of the generated airfoil
    print(f"Generated airfoil shape: {gen_coords.shape}")
    return gen_coords

def plot_airfoils(coords, cl, save_path="generated_airfoils.png"):
    """
    Plot and save airfoil shapes.

    Args:
        coords (np.ndarray): Airfoil coordinates (1, 1, 496) to be reshaped to (N, 496).
        cl (float): Target lift coefficient (C_L).
        save_path (str): File path to save the plot.
    """
    # Reshape coords to remove extra dimensions
    coords = np.asarray(coords)
    if len(coords.shape) == 3 and coords.shape[1] == 1:
        coords = coords.reshape(-1, 496)  # Reshape to (N, 496)

    # Validate shape
    if len(coords.shape) != 2 or coords.shape[1] != 496:
        raise ValueError(f"Expected coords to have shape (N, 496), but got {coords.shape}")

    num_airfoils = coords.shape[0]
    fig, ax = plt.subplots(1, num_airfoils, figsize=(15, 5), sharex=True, sharey=True)

    # Ensure ax is iterable, even if there's only one subplot
    if num_airfoils == 1:
        ax = [ax]

    for i, coord in enumerate(coords):
        # Split the 496-point vector into x and y
        x = coord[:248].flatten()  # First 248 points
        y = coord[248:].flatten()  # Last 248 points

        # Debugging output
        print(f"Airfoil {i}: x.shape={x.shape}, y.shape={y.shape}")
        # Print x values
        print(f"Airfoil {i}: x={x}")
        # Print y values
        print(f"Airfoil {i}: y={y}")

        # Validate dimensions
        if len(x) != 248 or len(y) != 248:
            raise ValueError(f"Coordinate dimensions are invalid: x({len(x)}), y({len(y)})")

        ax[i].plot(x, y)
        ax[i].set_title(f'C_L={cl:.3f}')
        ax[i].set_xlim(-0.1, 1.1)  # Adjust as needed
        ax[i].set_ylim(-0.5, 0.5)  # Adjust as needed

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Airfoils saved to {save_path}")



if __name__ == "__main__":
    # Fixed path for the pre-trained generator model
    model_path = "results/generator_params_45000"

    parser = argparse.ArgumentParser(description="Generate airfoil shapes using a trained GAN model.")
    parser.add_argument("--cl", type=float, required=True, help="Target lift coefficient (C_L).")
    parser.add_argument("--output_path", type=str, default="generated_airfoils.png", help="Path to save the generated airfoil plots.")
    parser.add_argument("--data_num", type=int, default=1, help="Number of airfoils to generate.")
    args = parser.parse_args()

    # Load the generator
    generator = load_generator(model_path)

    # Generate airfoils
    coords = generate_airfoil(generator, cl=args.cl, data_num=args.data_num)

    # Save and plot the results
    plot_airfoils(coords, cl=args.cl, save_path=args.output_path)
