if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from gan.models import Generator
import matplotlib.pyplot as plt
from calc_cl import get_cl, get_cls
from gan.utils import to_cpu, to_cuda, save_coords_by_cl 

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Eval:
  def __init__(self, G_PATH, coords_npz):
    state_dict = torch.load(G_PATH, map_location=torch.device('cpu'))
    self.G = Generator(3)
    self.G.load_state_dict(state_dict)
    self.G.eval()
    self.latent_dim = 3
    self.coords = {
      'data': coords_npz[coords_npz.files[0]],
      'mean':coords_npz[coords_npz.files[1]],
      'std':coords_npz[coords_npz.files[2]],
    }

  def rev_standardize(self, coords):
    return coords*self.coords['std']+self.coords['mean']

  def create_coords_by_cl(self, cl_c, data_num=20):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, self.latent_dim))))
    labels = np.array([cl_c]*data_num)
    labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, 1)))
    gen_coords = self.rev_standardize(to_cpu(self.G(z, labels)).detach().numpy())
    return gen_coords

  def create_successive_coords(self):
    """Generate 151 airfoil shapes with C_L^c ranging from 0.01 to 1.50"""
    cl_r = []
    cl_c = []
    gen_coords = []
    for cl in range(151):
      cl /= 100
      cl_c.append(cl)
      labels = Variable(torch.reshape(FloatTensor([cl]), (1, 1)))
      calc_num = 0
      while (True):
        calc_num += 1
        z = Variable(FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
        gen_coord = self.rev_standardize(to_cpu(self.G(z, labels)).detach().numpy())
        clr = get_cl(gen_coord)
        # cl = 0.1
        if not np.isnan(clr):
          print(cl)
          cl_r.append(clr)
          gen_coords.append(gen_coord)
          break
        if calc_num == 5:
          print('not calculated {0}'.format(cl))
          cl_r.append(-1)
          gen_coords.append(gen_coord)
          break

    np.savez("gan/results/successive_label", cl_c, cl_r, gen_coords)

  def save_coords(self, gen_coords, labels, path):
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = gen_coords[i]
        label = labels[i]
        x,y = coord.reshape(2, -1)
        ax[i%4, i//4].plot(x,y)
        cl = round(label.item(), 4)
        title = 'CL={0}'.format(str(cl))
        ax[i%4, i//4].set_title(title)

    fig.savefig(path)

  def successive(self):
    coords_npz = np.load("gan/results/successive_label.npz")
    cl_c = coords_npz[coords_npz.files[0]]
    cl_r = coords_npz[coords_npz.files[1]]
    success_clc = []
    success_clr = []
    fail_clc = []
    fail_clr = []
    for c, r in zip(cl_c, cl_r):
      if r == -1:
        fail_clc.append(c)
        fail_clr.append(0)
        continue
      success_clc.append(c)
      success_clr.append(r)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1.5])
    x = np.linspace(0, 1.5, 10)
    ax.plot(x, x, color = "black")
    ax.scatter(success_clc, success_clr)
    ax.scatter(fail_clc, fail_clr, color='red')
    ax.set_xlabel("Specified label")
    ax.set_ylabel("Recalculated label")
    # plt.show()
    fig.savefig("gan/results/successive_label.png")

  def sample_data(self, data_num=100):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, 3))))
    labels = 1.558*np.random.random_sample(size=(data_num, 1))
    labels = Variable(FloatTensor(labels))
    gen_coords = to_cpu(self.G(z, labels)).detach().numpy()
    labels = to_cpu(labels).detach().numpy()
    np.savez("gan/results/final", labels,self.rev_standardize(gen_coords))

  def euclid_dist(self, coords):
    """バリエーションがどれぐらいあるか"""
    mean = np.mean(coords, axis=0)
    mu_d = np.linalg.norm(coords - mean)/len(coords)
    return mu_d

  def _dist_from_dataset(self, coord):
    """データセットからの距離の最小値"""
    min_dist = 100
    idx = -1
    for i, data in enumerate(self.rev_standardize(self.coords['data'])):
      dist = np.linalg.norm(coord - data)
      if dist < min_dist:
        min_dist = dist
        idx = i
    
    return min_dist, idx
    
  def calc_dist_from_dataset(self, coords, clr):
    data_idx = -1
    generate_idx = -1
    max_dist = 0
    for i, c in enumerate(coords):
      cl = clr[i]
      if not np.isnan(cl):
        dist, didx = self._dist_from_dataset(c)
        if dist > max_dist:
          max_dist = dist
          data_idx = didx
          generate_idx = i
    return max_dist, data_idx, generate_idx
  
  def calc_mse(self):
        """
        Calcule le MSE entre les labels spécifiés et les labels recalculés.
        """
        coords_npz = np.load("gan/results/successive_label.npz")
        specified_labels = coords_npz[coords_npz.files[0]]  # Labels spécifiés
        recalculated_labels = coords_npz[coords_npz.files[1]]  # Labels recalculés

        # Filtrer les labels valides (éviter -1)
        valid_indices = recalculated_labels != -1
        specified_labels = specified_labels[valid_indices]
        recalculated_labels = recalculated_labels[valid_indices]

        # Calcul du MSE
        mse = np.mean((specified_labels - recalculated_labels) ** 2)

        print(f"MSE entre les labels spécifiés et recalculés : {mse:.6f}")
        return mse

if __name__ == "__main__":

    coords_npz = np.load("dataset/standardized_coords.npz")
    perfs = np.load("dataset/perfs.npy")
    G_PATH = "gan/results/generator_params_45000"
    evl = Eval(G_PATH, coords_npz)

    # List of target lift coefficients
    cl_values = [0.5, 1.0, 1.2]

    # Generate profiles for each CL value
    for cl_c in cl_values:
        print(f"Generating profiles for CL = {cl_c}")
        
        # Generate 12 profiles for the given CL
        data_num = 12
        coords = evl.create_coords_by_cl(cl_c, data_num=data_num)
        coords = coords.reshape(coords.shape[0], -1)

        # Calculate the average Euclidean distance
        mu = evl.euclid_dist(coords)
        print(f"Average distance for CL = {cl_c}: {mu}")

        # Display the 12 profiles in a 4x3 grid
        fig, axes = plt.subplots(4, 3, figsize=(12, 8))  # Create a 4x3 grid
        fig.suptitle(f"cGAN Generated profiles for CL = {cl_c}", fontsize=16)

        for i, coord in enumerate(coords):
            ax = axes[i // 3, i % 3]  # Select the axis in the grid
            x, y = coord.reshape(2, -1)
            ax.plot(x, y)
            ax.set_title(f"Profile {i+1}", fontsize=10)
            ax.axis('equal')  # Equal proportions for each axis
            ax.grid(True)

        # Remove unused subplots (if there are fewer than 12)
        for j in range(data_num, 12):
            fig.delaxes(axes[j // 3, j % 3])

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust margins around the main title

        # Save the generated profiles to a file
        output_file = f"gan/results/generated_profiles_{cl_c}.png"
        plt.savefig(output_file)
        print(f"Saved generated profiles to {output_file}")
        # plt.show()


  # evl.create_successive_coords()
  # evl.successive()
  # mse = evl.calc_mse()
  # print(mse)
  # clr = get_cls(coords)
  # max_dist, d_idx, g_idx = evl.calc_dist_from_dataset(coords, clr)
  # print(max_dist)
  # d_coord = evl.rev_standardize(evl.coords['data'][d_idx])
  # d_cl = perfs[d_idx]
  # g_coord = coords[g_idx]
  # g_cl = clr[g_idx]
  # print(cl_c, d_cl, g_cl)
  # cls = np.array([cl_c, d_cl, g_cl])
  # np.savez("dist_{0}".format(cl_c), d_coord, g_coord, cls, max_dist)
