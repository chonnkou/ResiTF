# Resilient Team Formation through Prescriptive Hypergraph Generation

This repository contains the core implementation and experiment notebooks for a working paper on resilient team formation. A team assignment is represented as a hypergraph: agents are nodes, tasks are hyperedges, and node–edge incidences encode assignments.

The proposed construction creates feasible, connected assignments and uses residual agent capacity to improve resilience. The repository also includes the search- and generation-based baselines used for method validation.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── examples/
│   └── quickstart.py
├── src/
│   ├── generation.py
│   ├── greedy_repair.py
│   ├── simulated_annealing.py
│   └── simulated_annealing_mu2.py
└── notebooks/
    ├── model_formulation.ipynb
    ├── experiment_workflow.ipynb
    ├── figures_validation.ipynb
    ├── figures_connectivity_and_search.ipynb
    ├── figures_ablation_and_sensitivity.ipynb
    └── figures_supplementary.ipynb
```

## Main components

- `src/generation.py`: proposed hypergraph construction, ablation variants, resilience measures, ER/BA baselines, experiment runners, and JSON result writers.
- `src/simulated_annealing.py`: constrained simulated-annealing benchmark.
- `src/simulated_annealing_mu2.py`: algebraic-connectivity-guided simulated-annealing benchmark.
- `src/greedy_repair.py`: greedy-repair benchmark.
- `notebooks/model_formulation.ipynb`: mathematical formulation and method description.
- `notebooks/experiment_workflow.ipynb`: entry points for validation, benchmarks, runtime, ablation, and sensitivity experiments.
- `notebooks/figures_*.ipynb`: plotting code for the paper's validation and robustness analyses.

## Installation

The code was run with Python 3.11. Create a clean environment and install the recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

Run the included small example from the repository root:

```bash
PYTHONPATH=src python examples/quickstart.py
```

Equivalent Python usage:

```python
import generation as gen

N, T, B, R = 12, 8, 3, 3
H = gen.gene_fcr(N, T, B, R)

print(H.shape)
print(gen.H_deg_edge_info(H))
print(gen.algebraic_connectivity(gen.H2X(H)))
```

The parameters must satisfy `N * B >= T * R`; otherwise the assignment is infeasible under the homogeneous budget and requirement assumptions.

## Reproducing experiments

1. Start Jupyter from the repository root with `jupyter notebook` or `jupyter lab`.
2. Open `notebooks/experiment_workflow.ipynb` and select the experiment block to run.
3. Generated JSON files are written to `results_*` directories according to the experiment type.
4. Use the matching `notebooks/figures_*.ipynb` notebook to create figures from those results.

The notebooks contain the parameter configurations used during development. Most long-running calls are commented out intentionally so that opening a notebook does not launch a large experiment. Set both `random.seed(...)` and `numpy.random.seed(...)` before an experiment when exact stochastic replication is required.



