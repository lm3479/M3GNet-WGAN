import os 
import numpy as np
from numpy import random
from typing import Tuple
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Conv3DTranspose, Dropout, Input, Layer, InputSpec
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU, BatchNormalization, Reshape, LayerNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import InputSpec
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import Model
import tensorflow_addons as tfa
import math
import argparse
from sklearn.cluster import KMeans
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure, Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet

def load_real_samples(data_path: str) -> np.ndarray:
    data_tensor = np.load(data_path)
    return np.reshape(data_tensor, (data_tensor.shape[0], 64, 64, 4))

def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in
        model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes

def conv_norm(x: tf.Tensor, units: int,
              filter: Tuple[int, int], stride: Tuple[int, int],
              discriminator: bool = True
              ) -> tf.Tensor:
  if discriminator:
    conv = Conv3D(units, filter, strides = stride, padding = 'valid')
  else:
    conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
  x = tfa.layers.SpectralNormalization(conv)(x)
  x = LayerNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  return x

def dense_norm(x: tf.Tensor, units: int
               ) -> tf.Tensor:
  x = tfa.layers.SpectralNormalization(Dense(units))(x)
  x = LayerNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  return x

#Normalization Constants
direction_fix = 0.9999933
normalize_avg_std = {'atom': [27.503517675002012, 22.417031299395962, 1.1822938247722463, 3.189382276810422],
                        'x': [0.4252026276473906, 0.2934632568241515, 1.4489126586030487, 1.9586462438477188],
                        'y': [0.43638668432502076, 0.291906737713482, 1.4949524212536385, 1.9307764302498376],
                        'z': [0.4442944632355303, 0.2930835040637591, 1.515931320170363, 1.8960314349305054],
                        'a': [7.576401996204324, 2.9769371381202365, 1.9530147015047585, 11.78647595694663],
                        'b': [7.90142069082381, 3.559553637232651, 1.724660257008108, 10.47366695621996],
                        'c': [9.930851747522944, 6.749988198167417, 1.150054121521355, 15.742864303277916],
                        'alpha': [89.9627072033887, 3.1373304207017987, 9.624649990367965, 9.574156613663968],
                        'beta': [93.78688773759445, 9.618284407608558, 3.543759603389515, 4.640756123524075],
                        'gamma': [95.65975023932918, 12.237549174937287, 3.0022964332249518, 1.9889807520054794],
                        'sg': [99.50188767306616, 76.65640490497339, 1.284979223786628, 1.7023771528120184],
                        'dir': [-4.058145803781749e-20, 0.4099350434569409, 2.4393946048146806, 2.4393946048146806],
                        'length': [0.6693464353927602, 0.236887249528413, 2.73534326062805, 4.159475792980244]}

#Crystal parameters as they appear in CrysTens
attribute_list = ["atom", "x", "y", "z", "a", "b", "c", "alpha", "beta", "gamma", "sg"]
parameter_list = ["a", "b", "c", "alpha", "beta", "gamma", "sg"]


crys_tens_path = r"C:\Users\trupt\Documents\Lavanya\Code\test.npy"
cif_folder = r"C:\Users\trupt\Documents\Lavanya\Code\cif_folder"
stats_folder = r"C:\Users\trupt\Documents\Lavanya\Code\stats_folder"

def unnormalize_crys_tens(crys_tens):
  crys_tens = np.copy(crys_tens)

  for site_idx in range(crys_tens.shape[0] - 12):

    #Check if an atom is present
    if np.any(crys_tens[12 + site_idx, :, :]):
      for att_idx, att in enumerate(attribute_list):
        crys_tens[att_idx, 12 + site_idx, :] *= normalize_avg_std[att][2] + normalize_avg_std[att][3]
        crys_tens[att_idx, 12 + site_idx, :] -= normalize_avg_std[att][2]
        crys_tens[att_idx, 12 + site_idx, :] *= normalize_avg_std[att][1]
        crys_tens[att_idx, 12 + site_idx, :] += normalize_avg_std[att][0]
        
        crys_tens[12 + site_idx, att_idx, :] *= normalize_avg_std[att][2] + normalize_avg_std[att][3]
        crys_tens[12 + site_idx, att_idx, :] -= normalize_avg_std[att][2]
        crys_tens[12 + site_idx, att_idx, :] *= normalize_avg_std[att][1]
        crys_tens[12 + site_idx, att_idx, :] += normalize_avg_std[att][0] 

      for adj_idx in range(crys_tens.shape[0] - 12):
        if site_idx != adj_idx:
          crys_tens[12 + site_idx, 12 + adj_idx, 0] *= normalize_avg_std["length"][2] + normalize_avg_std["length"][3]
          crys_tens[12 + site_idx, 12 + adj_idx, 0] -= normalize_avg_std["length"][2]
          crys_tens[12 + site_idx, 12 + adj_idx, 0] *= normalize_avg_std["length"][1]
          crys_tens[12 + site_idx, 12 + adj_idx, 0] += normalize_avg_std["length"][0]

          crys_tens[12 + site_idx, 12 + adj_idx, 1:] *= normalize_avg_std["dir"][2] + normalize_avg_std["dir"][3]
          crys_tens[12 + site_idx, 12 + adj_idx, 1:] -= normalize_avg_std["dir"][2]
          crys_tens[12 + site_idx, 12 + adj_idx, 1:] *= normalize_avg_std["dir"][1]
          crys_tens[12 + site_idx, 12 + adj_idx, 1:] += normalize_avg_std["dir"][0]

  return crys_tens

from typing import List
#Below is the ORIGINAL get_cif function copy pasted from repo 
def get_cif(crys_tens, file_path, ref_percent: float = 0.2, rel_coord_percent: float = None, num_coord_clusters: int = None, num_atom_clusters: int = 3, generator: str = "Real CIF", dir_diff_avg: List = None, pot_sites = None, top_sites: int = None):
    dir_diff_latest = []
    crystal_cif = {}
    ref_angle = crys_tens[12, 7, 0]
    ref_avg = ref_angle

    #Uses ref_percent to determine when there are no atoms left in CrysTens (column/row[0] = 0)
    for num in range(13, crys_tens.shape[0]):
      if abs(crys_tens[num, 7, 0] - ref_avg)/ref_avg >= ref_percent:
        break
      else:
        ref_angle += crys_tens[num, 7, 0]
        reference_average = ref_angle / (num - 11)
    
    #Averages each value with its reflected value
    constrained_crys_tens = crys_tens[:num, :num, :]
    for i in range(12, constrained_crys_tens.shape[0]):
      for j in range(12):
        avg_val = (np.sum(constrained_crys_tens[i, j, :]) + np.sum(constrained_crys_tens[j, i, :]))/(2 * constrained_crys_tens.shape[2]) 
        constrained_crys_tens[i, j, :], constrained_crys_tens[j, i, :] = avg_val, avg_val

    #Finds the average of parameters, angles, and space group number to use as the final value
    for i in range(4, 11):
      sum_val = 0
      for j in range(12, constrained_crys_tens.shape[0]):
        sum_val += np.sum(constrained_crys_tens[i, j, :]) + np.sum(constrained_crys_tens[j, i, :])
      avg_val = sum_val / (2 * (constrained_crys_tens.shape[0] - 12) * constrained_crys_tens.shape[2])
      constrained_crys_tens[i, 12:, :], constrained_crys_tens[12:, i, :] = avg_val, avg_val
      crystal_cif[parameter_list[i - 4]] = avg_val

    #Makes an attempt to symmetrize the CrysTens
    for i in range(12, constrained_crys_tens.shape[0]):
      for j in range(12, constrained_crys_tens.shape[0]):
        avg_val = (constrained_crys_tens[i, j, 0] + constrained_crys_tens[j, i, 0])/2
        constrained_crys_tens[i, j, 0], constrained_crys_tens[j, i, 0] = avg_val, avg_val
        for k in range(1, 4):
          if constrained_crys_tens[i, j, k] > 0:
            avg_val = (constrained_crys_tens[i, j, k] + abs(constrained_crys_tens[j, i, k]))/2
            constrained_crys_tens[i, j, k], constrained_crys_tens[j, i, k] = avg_val, -avg_val
          else:
            avg_val = (abs(constrained_crys_tens[i, j, k]) + constrained_crys_tens[j, i, k])/2
            constrained_crys_tens[i, j, k], constrained_crys_tens[j, i, k] = -avg_val, avg_val

    #Finds the absolute x, y, and z coordinates of each atom as well as the relative coordinates of each atom (according to the distance matrices)
    crystal_cif["site_list"] = {}
    for i in range(12, constrained_crys_tens.shape[0]):
      crystal_cif["site_list"][i - 12] = {}
      crystal_cif["site_list"][i - 12]["atom"] = constrained_crys_tens[i, 0, 0]
      crystal_cif["site_list"][i - 12]["x"] = constrained_crys_tens[i, 1, 0]
      crystal_cif["site_list"][i - 12]["y"] = constrained_crys_tens[i, 2, 0]
      crystal_cif["site_list"][i - 12]["z"] = constrained_crys_tens[i, 3, 0]

      crystal_cif["site_list"][i - 12]["adj_list"] = []

    for i in range(12, constrained_crys_tens.shape[0]):
      for j in range(12, constrained_crys_tens.shape[0]):
        adj_x = crystal_cif["site_list"][i - 12]["x"] - constrained_crys_tens[i, j, 1]
        adj_y = crystal_cif["site_list"][i - 12]["y"] - constrained_crys_tens[i, j, 2]
        adj_z = crystal_cif["site_list"][i - 12]["z"] - constrained_crys_tens[i, j, 3]
        crystal_cif["site_list"][j - 12]["adj_list"].append((adj_x, adj_y, adj_z))

    site_list = []
    atom_list = []
    if rel_coord_percent is None:
      rel_coord_percent = 1 - (1/constrained_crys_tens.shape[0])
    
    #Calculates the final coordinate positions and calculates the difference between the final positions and the relative positions
    for site_idx in crystal_cif["site_list"]:
      site = crystal_cif["site_list"][site_idx]
      site_x = site["x"]
      site_y = site["y"]
      site_z = site["z"]
      adj_x_list = []
      adj_y_list = []
      adj_z_list = []
      adj_coord_list =site["adj_list"]
      for adj_idx in range(len(adj_coord_list)):
        adj_coord = adj_coord_list[adj_idx]
        adj_x_list.append(adj_coord[0])
        adj_y_list.append(adj_coord[1])
        adj_z_list.append(adj_coord[2])
      
      site_coord_percent = 1 - rel_coord_percent
      x_rel = site_coord_percent*site_x + rel_coord_percent*np.average(adj_x_list)
      y_rel = site_coord_percent*site_y + rel_coord_percent*np.average(adj_y_list)
      z_rel = site_coord_percent*site_z + rel_coord_percent*np.average(adj_z_list)
      atom_list.append(np.around(site["atom"]))
      site_list.append((x_rel, y_rel, z_rel))
      if dir_diff_avg is not None:
        dir_diff_latest.append(abs(site_x - x_rel)/len(adj_x_list))
        dir_diff_latest.append(abs(site_y - y_rel)/len(adj_y_list))
        dir_diff_latest.append(abs(site_z - z_rel)/len(adj_z_list))

    #Reconstructs a new pairwise distance matrix and compares it to the one generated in CrysTens Layer 1
    reconstructed_pairwise = np.zeros((constrained_crys_tens.shape[0] - 12, constrained_crys_tens.shape[1] - 12, 1))
    for site_idx in range(len(site_list)):
      site = site_list[site_idx]
      for adj_idx in range(len(site_list)):
        adj = site_list[adj_idx]
        x_diff = site[0] - adj[0]
        y_diff = site[1] - adj[1]
        z_diff = site[2] - adj[2]
        length = (x_diff**2 + y_diff**2 + z_diff**2)**(1/2)

        reconstructed_pairwise[site_idx, adj_idx, 0] = length
        reconstructed_pairwise[adj_idx, site_idx, 0] = length
    
    pairwise_error = np.abs(constrained_crys_tens[12:, 12:, 0] - reconstructed_pairwise[:, :, 0])

    #K-Means Clustering for the coordinate values
    dir_diff_avg.append(dir_diff_latest)
    kmeans_coord_list = []
    for i in range(len(site_list)):
      for j in range(3):
        kmeans_coord_list.append(site_list[i][j])

    if num_coord_clusters is None:
      num_coord_clusters = len(kmeans_coord_list)
    
    num_coord_clusters = min(len(kmeans_coord_list), num_coord_clusters)

    kmeans_coord = KMeans(n_clusters = num_coord_clusters).fit(np.array(kmeans_coord_list).reshape(-1, 1))
    for i in range(len(kmeans_coord_list)):
      kmeans_coord_list[i] = kmeans_coord.cluster_centers_[kmeans_coord.labels_[i]][0]
    
    for i in range(0, len(kmeans_coord_list), 3):
      site_list[i//3] = [kmeans_coord_list[i], kmeans_coord_list[i + 1], kmeans_coord_list[i + 2]]
    
    
    #TODO Add PotScoring

    #K-Means Clustering for the atom values
    if num_atom_clusters is None:
      num_atom_clusters = len(atom_list)
    
    num_atom_clusters = min(num_atom_clusters, len(atom_list))
    kmeans_atom = KMeans(n_clusters=int(num_atom_clusters)).fit(np.array(atom_list).reshape(-1, 1))

    for i in range(len(atom_list)):
      atom_list[i] = np.around(kmeans_atom.cluster_centers_[kmeans_atom.labels_[i]])[0]

    #Remove duplicates
    trimmed_site_list = []
    trimmed_atom_list = []
    dup_check = set()
    for i in range(len(atom_list)):
      if (atom_list[i], tuple(site_list[i])) not in dup_check:
        dup_check.add((atom_list[i], tuple(site_list[i])))
        trimmed_site_list.append(site_list[i])
        trimmed_atom_list.append(atom_list[i])

    #Creates the CIF
    lattice = Lattice.from_parameters(a = crystal_cif["a"], b = crystal_cif["b"], c = crystal_cif["c"], alpha = crystal_cif["alpha"], beta = crystal_cif["beta"], gamma = crystal_cif["gamma"])
    struct = Structure(lattice = lattice, species = trimmed_atom_list, coords = trimmed_site_list, to_unit_cell=True)
    written_cif = str(CifWriter(struct))
    with open(file_path, "w") as file:
      file.write("Generated by: " + generator + "\n" + "Num unique sites: " + str(num_coord_clusters) + "\n" + "Num unique elements: " + str(num_atom_clusters) + "\n\n" + written_cif)
    
    return crystal_cif["sg"], np.sum(pairwise_error)/len(site_list)

# below is the MODIFIED get_cif function that gets_cif instead" 


def get_structure(y_pred, ref_percent: float = 0.2, rel_coord_percent: float = None, 
                  num_coord_clusters: int = None, num_atom_clusters: int = 3, dir_diff_avg: list = None):
    crys_tens = np.squeeze(y_pred)
    dir_diff_latest = []
    crystal_cif = {}
    ref_angle = crys_tens[12, 7, 0]
    ref_avg = ref_angle

    for num in range(13, crys_tens.shape[0]):
        if abs(crys_tens[num, 7, 0] - ref_avg) / ref_avg >= ref_percent:
            break
        else:
            ref_angle += crys_tens[num, 7, 0]
            ref_avg = ref_angle / (num - 11)

    constrained_crys_tens = crys_tens[:num, :num, :]
    for i in range(12, constrained_crys_tens.shape[0]):
        for j in range(12):
            avg_val = (np.sum(constrained_crys_tens[i, j, :]) + np.sum(constrained_crys_tens[j, i, :])) / (2 * constrained_crys_tens.shape[2])
            constrained_crys_tens[i, j, :], constrained_crys_tens[j, i, :] = avg_val, avg_val

    for i in range(4, 11):
        sum_val = 0
        for j in range(12, constrained_crys_tens.shape[0]):
            sum_val += np.sum(constrained_crys_tens[i, j, :]) + np.sum(constrained_crys_tens[j, i, :])
        avg_val = sum_val / (2 * (constrained_crys_tens.shape[0] - 12) * constrained_crys_tens.shape[2])
        crystal_cif[parameter_list[i - 4]] = avg_val

    site_list = []
    atom_list = []
    if rel_coord_percent is None:
        rel_coord_percent = 1 - (1 / constrained_crys_tens.shape[0])

    for i in range(12, constrained_crys_tens.shape[0]):
        site_x = constrained_crys_tens[i, 1, 0]
        site_y = constrained_crys_tens[i, 2, 0]
        site_z = constrained_crys_tens[i, 3, 0]
        atom_list.append(np.around(constrained_crys_tens[i, 0, 0]))
        site_list.append((site_x, site_y, site_z))

    trimmed_site_list = []
    trimmed_atom_list = []
    dup_check = set()
    for i in range(len(atom_list)):
        if (atom_list[i], tuple(site_list[i])) not in dup_check:
            dup_check.add((atom_list[i], tuple(site_list[i])))
            trimmed_site_list.append(site_list[i])
            trimmed_atom_list.append(atom_list[i])

    lattice = Lattice.from_parameters(
        a=crystal_cif["a"], b=crystal_cif["b"], c=crystal_cif["c"],
        alpha=crystal_cif["alpha"], beta=crystal_cif["beta"], gamma=crystal_cif["gamma"]
    )
    struct = Structure(lattice=lattice, species=trimmed_atom_list, coords=trimmed_site_list, to_unit_cell=True)
    
    return struct

def get_cif_symbolic(
    crys_tens, file_path = "", ref_percent=0.2, rel_coord_percent=None, 
    num_coord_clusters=None, num_atom_clusters=3, generator="Real CIF"
):
    print("CHEKPT 1.-1")
    dir_diff_latest = []
    crys_tens = unnormalize_crys_tens(crys_tens)
    ref_angle = crys_tens[12, 7, 0]
    ref_avg = ref_angle
    print("CHEKPT 1.0")
    # Calculate the reference average using symbolic operations
    condition = tf.abs(crys_tens[13:, 7, 0] - ref_avg) / ref_avg < ref_percent
    valid_indices = tf.where(condition)[:, 0] + 13  # Indices of valid rows
    ref_values = tf.gather(crys_tens[:, 7, 0], valid_indices)
    ref_angle = tf.reduce_sum(ref_values)
    reference_average = ref_angle / tf.cast(tf.shape(ref_values)[0], tf.float32)

    print("CHEKPT 1.1")
    # Slice the constrained crystal tensor
    num = tf.reduce_max(valid_indices) + 1
    constrained_crys_tens = crys_tens[:num, :num, :]

    # Average each value with its reflected value
    upper_triangle = tf.linalg.band_part(constrained_crys_tens, 0, -1)
    lower_triangle = tf.linalg.band_part(constrained_crys_tens, -1, 0)
    avg_tensor = (upper_triangle + tf.transpose(lower_triangle, perm=[1, 0, 2])) / 2
    constrained_crys_tens = avg_tensor
    print("CHEKPT 1.2")
    # Find averages for parameters, angles, and space group number
    crystal_cif = {}
    for i in range(4, 11):
        sum_vals = tf.reduce_sum(constrained_crys_tens[i, 12:, :], axis=1)
        avg_val = sum_vals / tf.cast(num - 12, tf.float32)
        crystal_cif[f"parameter_{i-4}"] = avg_val

    # Symmetrize the tensor
    upper_triangle = tf.linalg.band_part(constrained_crys_tens, 0, -1)
    lower_triangle = tf.linalg.band_part(constrained_crys_tens, -1, 0)
    constrained_crys_tens = (upper_triangle + tf.transpose(lower_triangle, perm=[1, 0, 2])) / 2
    print("CHEKPT 1.3")
    # Calculate the final coordinates
    if rel_coord_percent is None:
        rel_coord_percent = 1 - (1 / tf.cast(num, tf.float32))

    site_list = []
    atom_list = []
    num = tf.cast(num, dtype=tf.int32)
    for i in range(12, num):
        site_x = constrained_crys_tens[i, 1, 0]
        site_y = constrained_crys_tens[i, 2, 0]
        site_z = constrained_crys_tens[i, 3, 0]
        adj_list = constrained_crys_tens[i, 12:num, 1:4]
        adj_avg = tf.reduce_mean(adj_list, axis=0)
        x_rel = (1 - rel_coord_percent) * site_x + rel_coord_percent * adj_avg[0]
        y_rel = (1 - rel_coord_percent) * site_y + rel_coord_percent * adj_avg[1]
        z_rel = (1 - rel_coord_percent) * site_z + rel_coord_percent * adj_avg[2]
        site_list.append([x_rel, y_rel, z_rel])
        atom_list.append(constrained_crys_tens[i, 0, 0])
    

    def kmeans_clustering(data, num_clusters, num_iterations=10):
      """
      TensorFlow-based KMeans clustering.
      """
      centroids = tf.gather(data, tf.random.shuffle(tf.range(tf.shape(data)[0]))[:num_clusters])

      for _ in range(num_iterations):
          distances = tf.reduce_sum(tf.square(tf.expand_dims(data, 1) - tf.expand_dims(centroids, 0)), axis=2)
          assignments = tf.argmin(distances, axis=1)
          centroids = tf.stack([
              tf.reduce_mean(tf.boolean_mask(data, assignments == k), axis=0)
              for k in range(num_clusters)
          ], axis=0)
      return centroids, assignments

    # Perform K-Means clustering (requires concrete tensors for clustering)
    atom_list_np = tf.expand_dims(atom_list, axis=1) # Convert to NumPy for clustering
    clustered_centroids, cluster_assignments = kmeans_clustering(atom_list_np, num_atom_clusters)
    clustered_atoms = tf.gather(clustered_centroids, cluster_assignments)

    site_list_flat = tf.reshape(site_list, [-1])
    concatenated = tf.concat([site_list_flat, tf.squeeze(clustered_centroids)], axis=0)
    unique_sites, idx = tf.unique(concatenated)

    print("Post Kmeans")

    # Remove duplicates
    concatenated = tf.concat([tf.reshape(site_list, [-1]), tf.squeeze(clustered_atoms)], axis=0)
    unique_sites, idx = tf.unique(concatenated)
    print("CHEKPT 1.4")
    # Create the CIF
    lattice = Lattice.from_parameters(
        a=float(np.mean(crystal_cif["parameter_0"])),
        b=float(np.mean(crystal_cif["parameter_1"])),
        c=float(np.mean(crystal_cif["parameter_2"])),
        alpha=float(np.mean(crystal_cif["parameter_3"])),
        beta=float(np.mean(crystal_cif["parameter_4"])),
        gamma=float(np.mean(crystal_cif["parameter_5"]))
    )
    print(clustered_atoms.numpy().shape)
    print(clustered_atoms)
    struct = Structure(
        lattice=lattice, species=clustered_atoms.numpy().tolist(),
        coords=np.array(site_list), to_unit_cell=True
    )
    print("CHEKPT 1.5")
    return struct

import json
from typing import Optional
def from_dir(dirname: str, custom_objects: Optional[dict] = None):
    """
    Load the model from a directory

    Args:
        dirname (str): directory to save the model
        custom_objects (dict): dictionary for custom object
    Returns: M3GNet model
    """
    custom_objects = custom_objects or {}
    model_name = r"c:\Users\trupt\Documents\Lavanya\Code\M3GNetWeights\m3gnet"
    fname = os.path.join(dirname, 'm3gnet' + ".json")
    if not os.path.isfile(fname):
        raise ValueError("Model does not exists")
    with open(fname) as f:
        model_serialized = json.load(f)
    # model_serialized = _replace_compatibility(model_serialized)
    model = tf.keras.models.model_from_json(model_serialized, custom_objects=custom_objects)
    model.load_weights(model_name)
    print("WEIGHTS LOADED")
    return model

class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=512, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        return self

    def __next__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size, shape)

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1) #Am I supposed to add + score @Michael

def m3gnet_loss(_, y_pred):
    m3gnet_model_path = r"C:\Users\trupt\Documents\Lavanya\Code\M3GNetWeights\m3gnet"
    mp_api_key = 'l3TUAzXlkLkdSeimWMv8BNqBrJgxGHWB'
    y_pred = tf.squeeze(y_pred, axis=[0, -1])

    #m3gnet_e_form = from_dir(m3gnet_model_path)
    try:
        struct = get_cif_symbolic(y_pred)
        e_form_predict = m3gnet_e_form.predict_structure(struct)
        # print(f"Predicted formation energy for structure {i+1}: {e_form_predict}")
        elements = [el.symbol for el in struct.composition.elements]
        mpr = MPRester(mp_api_key)
        all_compounds = mpr.summary.search(elements=elements)
        pde_list = [ComputedEntry(c.composition, c.formation_energy_per_atom) for c in all_compounds]
            
        if not pde_list:
            raise ValueError(f"No valid phase diagram data for structure {i+1}")

        diagram = PhaseDiagram(pde_list)
        _, pmg_ehull = diagram.get_decomp_and_e_above_hull(
                ComputedEntry(struct.composition, e_form_predict)
            )
    except Exception as e:
        pmg_ehull = 5

       
    return pmg_ehull

class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x,y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1],0,1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights,-1)
        rw = x*weights + y*(1-weights)
        return rw

  class Nontrainable(object):

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.trainable_status = self.model.trainable
        self.model.trainable = False
        return self.model

    def __exit__(self, type, value, traceback):
        self.model.trainable = self.trainable_status

class GradientPenalty(Layer):
    def call(self, inputs):
        real_image, generated_image, disc = inputs
        avg_image = RandomWeightedAverage()(
        [real_image, generated_image]
        )
        with tf.GradientTape() as tape:
          tape.watch(avg_image)
          disc_avg = disc(avg_image)

        grad = tape.gradient(disc_avg,[avg_image])[0]
        GP = K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1
        return GP

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

  def generate_real_samples(dataset: np.ndarray, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    ix = random.randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]
    y = np.ones((n_samples,1))
    return X,y

def generate_latent_points(latent_dim: int, n_samples:int) -> np.ndarray:
    x_input = random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input

def generate_fake_samples(generator: tf.Tensor,
                          latent_dim: int, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    x_input = generate_latent_points(latent_dim,n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y

def define_critic(in_shape = (64, 64, 4, 1)
) -> tf.Tensor:
    tens_in = Input(shape=in_shape, name="input")

    x = conv_norm(tens_in, 16, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 16, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (5,5,2), (5,5,1), True)
    x = conv_norm(x, 128, (2,2,2), (2,2,2), True)

    x = Flatten()(x)
    x = Dropout(0.25)(x)

    disc_out = tfa.layers.SpectralNormalization(Dense(1, activation = "linear"))(x)
    model = Model(inputs=tens_in, outputs=disc_out)

    return model

def define_generator(latent_dim):
    n_nodes = 16 * 16 * 4

    noise_in = Input(shape=(latent_dim,), name="noise_input")

    x = dense_norm(noise_in, n_nodes)
    x = Reshape((16,16, 4, 1))(x)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (2,2,2), (2,2,2), False)

    outMat = tfa.layers.SpectralNormalization(Conv3D(1,(1,1,10), activation = 'sigmoid', strides = (1,1,10), padding = 'valid'))(x)

    model = Model(inputs=noise_in, outputs=outMat)
    return model

from tqdm import tqdm
class WGANGP(object):

        def __init__(self, gen, disc, lr_gen=0.0001, lr_disc=0.0001):

          self.gen = gen
          self.disc = disc
          self.lr_gen = lr_gen
          self.lr_disc = lr_disc
          self.build()







        def build(self):
            # ...
            try:
                #tens_shape = input_shapes(self.disc, "input")[0]
                tens_shape = (64, 64, 4, 1)
            except:
                tens_shape = (64, 64, 4, 1)
            try:
                noise_shapes = input_shapes(self.gen, "noise_input")
            except:
                noise_shapes =(128,)

            self.opt_disc = Adam(self.lr_disc, beta_1=0.0, beta_2=0.9)
            self.opt_gen = Adam(self.lr_gen, beta_1=0.0, beta_2=0.9)

            with Nontrainable(self.gen):
                real_image = Input(shape=tens_shape)
                noise = [Input(shape=s) for s in noise_shapes]

                disc_real = self.disc(real_image)
                generated_image = self.gen(noise)
                disc_fake = self.disc(generated_image)

                gp = GradientPenalty()([real_image, generated_image, self.disc])
                self.disc_trainer = Model(
                    inputs=[real_image, noise],
                    outputs=[disc_real, disc_fake, gp]
                )
                self.disc_trainer.compile(optimizer=self.opt_disc,
                    loss=[wasserstein_loss, wasserstein_loss, 'mse'],
                    loss_weights=[1.0, 1.0, 10.0]
                )

            with Nontrainable(self.disc):
                noise = [Input(shape=s) for s in noise_shapes]

                generated_image = self.gen(noise)
                disc_fake = self.disc(generated_image)

                self.gen_trainer = Model(
                    inputs=noise,
                    outputs=[disc_fake, generated_image]
                )
                self.gen_trainer.compile(optimizer=self.opt_gen,
                    loss=[wasserstein_loss, m3gnet_loss])



        def fit_generator(self, noise_gen, dataset, latent_dim, n_epochs=10, n_batch=20, n_critic=5, model_name=None):
          bat_per_epoch = int(1000 / n_batch)
          n_steps = bat_per_epoch * n_epochs
          half_batch = int(n_batch / 2)
          disc_out_shape = (n_batch, self.disc.output_shape[1])
          real_target = -np.ones(disc_out_shape, dtype=np.float32)
          fake_target = -real_target
          gp_target = np.zeros_like(real_target)
          lastEpoch = 0
          genLossArr = []
          disc0LossArr = []
          disc1LossArr = []
          disc2LossArr = []

          for epoch in range(n_epochs):
              print("Epoch {}/{}".format(epoch + 1, n_epochs))

              # Initialize tqdm progress bar
              progbar = tqdm(total=bat_per_epoch, desc=f'Epoch {epoch + 1}/{n_epochs}')

              for step in range(bat_per_epoch):
                  # Train discriminator
                  with Nontrainable(self.gen):
                      for repeat in range(n_critic):
                          tens_batch, _ = generate_real_samples(dataset, n_batch)
                          noise_batch = next(noise_gen)
                          disc_loss = self.disc_trainer.train_on_batch(
                              [tens_batch] + noise_batch,
                              [real_target, fake_target, gp_target]
                          )

                  # Train generator
                  with Nontrainable(self.disc):
                      noise_batch = next(noise_gen)
                      gen_loss = self.gen_trainer.train_on_batch(
                          noise_batch, real_target
                      )

                  losses = []
                  for i, dl in enumerate(disc_loss):
                      losses.append(("D{}".format(i), dl))
                      if i == 0:
                          disc0LossArr.append(dl)
                      elif i == 1:
                          disc1LossArr.append(dl)
                      elif i == 2:
                          disc2LossArr.append(dl)
                  losses.append(("G0", gen_loss))
                  genLossArr.append(gen_loss)
                  progbar.set_postfix(losses=dict(losses))
                  progbar.update(1)  # Update progress bar

              progbar.close()  # Close the progress bar after epoch

              # Save model and losses
              if model_name:
                  self.gen.save(model_name + "gen"+".keras")
                  self.disc.save(model_name + "critic"+".keras")
                  np.save(model_name + "real_loss"+".npy", np.array(disc0LossArr))
                  np.save(model_name + "fake_loss"+".npy", np.array(disc1LossArr))
                  np.save(model_name + "gp_loss"+".npy", np.array(disc2LossArr))
                  np.save(model_name + "generator_loss"+".npy", np.array(genLossArr))
                  print("Training complete!")
                
tf.config.run_functions_eagerly(True)  # For debugging

import os
batch_size = 1

data_path = './'
os.chdir(data_path)

file_path = 'test.npy'
data = np.load(file_path)

n_epochs = 15
n_critic = 5
model_path = 'model_out\\'

def main():
 #   args = parser.parse_args()
    noise_dim = 128
    critic = define_critic()
    generator = define_generator(noise_dim)
    gan_model = WGANGP(generator, critic)
    noise_gen = NoiseGenerator([noise_dim,], batch_size = batch_size)
    dataset = load_real_samples(file_path)
    gan_model.fit_generator(noise_gen, dataset, noise_dim, n_epochs, batch_size,n_critic, model_path)

if __name__ == "__main__":
    main()
