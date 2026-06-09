

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool, global_add_pool, Set2Set
import numpy as np
from torch_geometric.data import Data
import warnings
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pymatgen.core import Structure



def gaussian_expansion(distances, dmin=0, dmax=8, steps=100):
    filters = np.linspace(dmin, dmax, steps)
    sigma = 5 * (dmax - dmin) / steps  # filter spacing = 0.08 Å
    return np.exp(-((distances[:, None] - filters[None, :]) ** 2) / (sigma ** 2))


def get_atom_features_exp(site, n_species=2):
    species_sorted = []
    if site.is_ordered:
        elem = site.specie
        species_sorted = [(elem, 1.0)]
    else:
        species_sorted = sorted(site.species.items(), key=lambda x: x[1], reverse=True)

    features = []
    for i in range(n_species):
        if i < len(species_sorted):
            elem, occ = species_sorted[i]
            features += [
                float(elem.Z) / 94,
                float(occ),
                float(elem.X) if elem.X else 0.0,               # electronegativity
                float(elem.atomic_radius or 0.0),               # atomic radius (Å)
                float(elem.row) / 9,                            # period
                float(elem.group) / 18 if elem.group else 0.0, # group
            ]
        else:
            features += [0.0] * 6

    occupancies = [occ for _, occ in species_sorted]
    disorder = -sum(o * np.log(o + 1e-9) for o in occupancies)
    features.append(disorder)

    return features  # 2*6 + 1 = 13 features


def get_atom_features(site, n_species=2):
    species_sorted = []
    
    if site.is_ordered:
        elem = site.specie
        species_sorted = [(elem, 1.0)]
    else:
        species_sorted = sorted(site.species.items(), key=lambda x: x[1], reverse=True)

    features = []
    for i in range(n_species):
        if i < len(species_sorted):
            elem, occ = species_sorted[i]
            features.append(float(elem.Z))
            features.append(float(occ))
        else:
            features.append(0.0)
            features.append(0.0)

    # Site disorder (entropy)
    occupancies = [occ for _, occ in species_sorted]
    disorder = -sum(o * np.log(o + 1e-9) for o in occupancies)
    features.append(disorder)

    return features


def structure_to_graph(structure, target, weight = 1.0):
    # Node features
    a, b, c = structure.lattice.abc
    cutoff = 1.2 * min(a, b, c)
    atom_feats = np.array([get_atom_features_exp(site) for site in structure], dtype=np.float32)
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Find neighbors within cutoff
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    # --- adaptive neighbour cutoff: 1.2 × nearest-neighbour distance ---
    edge_src, edge_dst, edge_distances = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        if not neighbors:
            continue

        nearest_dist = min(nbr[1] for nbr in neighbors)
        adaptive_cutoff = 1.2 * nearest_dist

        for neighbor in neighbors:
            if neighbor[1] <= adaptive_cutoff:
                edge_src.append(i)
                edge_dst.append(neighbor[2])
                edge_distances.append(neighbor[1])

    if len(edge_src) == 0:
        return None  # skip pathological structures
    edge_index = torch.tensor(np.array([edge_src, edge_dst]), dtype=torch.long)

    # Edge features: Gaussian expanded distances
    distances = np.array(edge_distances)
    edge_attr = torch.tensor(gaussian_expansion(distances), dtype=torch.float)

    # Target value
    y = torch.tensor([target], dtype=torch.float)
    w = torch.tensor([weight], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w)
    return data


biased_split = True

data_ICSD = pd.read_csv( "3DSC_ICSD.csv", skiprows=1)


warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")
if True:
    graphs = []
    for i, row in data_ICSD.iterrows():
        print(f"{i}/{len(data_ICSD)}", end="\r")  # overwrites the same line
        try:
            structure = Structure.from_file(row["cif"].replace("C:\\Users\\rasmu\\OneDrive\\Skrivebord\\1-NytdropPython\\4.year\\AppliedML2026\\Examproject\\3DSC\\superconductors_3D\\data\\final\\ICSD\\cifs/","ICSD/"))
            if row['tc'] > 0.01:
                graph = structure_to_graph(structure, target=(np.log(row['tc'])-3.3)/2.5, weight=row['weight'])
                graphs.append(graph)
        except Exception as e:
            print(f"\nSkipping row {i}: {e}")

    print(f"Built {len(graphs)} graphs")
    print(graphs[0])  # inspect first graph
    torch.save(graphs, 'crystal_graphs_vClaude_ICSD_exp.pt')
if True:
    np.random.seed(42)
    # Group all row indices by formula
    formula_groups = data_ICSD[data_ICSD["tc"] > 0.01].groupby("formula_sc")

    # One entry per unique formula: representative tc + all row indices
    formulas      = np.array(list(formula_groups.groups.keys()))
    all_indices   = [formula_groups.groups[f].tolist() for f in formulas]
    tc_per_formula = np.array([formula_groups["tc"].mean()[f] for f in formulas])

    # Shuffle at the formula level
    perm = np.random.permutation(len(formulas))
    formulas    = formulas[perm]
    all_indices = [all_indices[i] for i in perm]

    # Split
    n_train = int(0.9 * len(formulas))

    train_row_idx = [idx for group in all_indices[:n_train] for idx in group]
    test_row_idx  = [idx for group in all_indices[n_train:] for idx in group]

    train_df = data_ICSD.loc[train_row_idx]
    test_df  = data_ICSD.loc[test_row_idx]

    graphs_val = []
    graphs_train = []
    for i, row in train_df.iterrows():
        print(f"{len(graphs_train)}/{len(train_df)}", end="\r")  # overwrites the same line
        try:
            structure = Structure.from_file(row["cif"].replace("C:\\Users\\rasmu\\OneDrive\\Skrivebord\\1-NytdropPython\\4.year\\AppliedML2026\\Examproject\\3DSC\\superconductors_3D\\data\\final\\ICSD\\cifs/","ICSD/"))
            if row['tc'] > 0.01:
                graph = structure_to_graph(structure, target=(np.log(row['tc'])-3.3)/2.5, weight=row['weight'])
                graphs_train.append(graph)
        except Exception as e:
            print(f"\nSkipping row {i}: {e}")

    print(f"Built {len(graphs_train)} graphs")
    print(graphs_train[0])  # inspect first graph
    torch.save(graphs_train, 'crystal_graphs_vClaude_ICSD_train_exp.pt')

    for i, row in test_df.iterrows():
        print(f"{len(graphs_val)}/{len(test_df)}", end="\r")  # overwrites the same line
        try:
            structure = Structure.from_file(row["cif"].replace("C:\\Users\\rasmu\\OneDrive\\Skrivebord\\1-NytdropPython\\4.year\\AppliedML2026\\Examproject\\3DSC\\superconductors_3D\\data\\final\\ICSD\\cifs/","ICSD/"))
            if row['tc'] > 0.01:
                graph = structure_to_graph(structure, target=(np.log(row['tc'])-3.3)/2.5, weight=row['weight'])
                graphs_val.append(graph)
        except Exception as e:
            print(f"\nSkipping row {i}: {e}")

    print(f"Built {len(graphs_val)} graphs")
    print(graphs_val[0])  # inspect first graph
    torch.save(graphs_val, 'crystal_graphs_vClaude_ICSD_val_exp.pt')






