"""Minimal example for the prescriptive team-formation method."""

import random

import numpy as np

import generation as gen


random.seed(7)
np.random.seed(7)

N, T, B, R = 12, 8, 3, 3
hypergraph = gen.gene_fcr(N, T, B, R)
incidence = gen.H2X(hypergraph)

print(f"Hypergraph shape (agents, tasks): {hypergraph.shape}")
print(f"Incidence matrix shape: {incidence.shape}")
print(f"Mean agent degree: {gen.mean_degree(hypergraph):.3f}")
print(f"Mean team size: {gen.mean_edges_size(hypergraph):.3f}")
print(f"Algebraic connectivity: {gen.algebraic_connectivity(incidence):.6f}")
