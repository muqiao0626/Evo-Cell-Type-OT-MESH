# OT-MESH: Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport

This repository implements OT-MESH (Optimal Transport with Minimize Entropy of Sinkhorn) for identifying evolutionary correspondences between cell types across species. The method is described in:

Qiao, M. (2025) Unsupervised Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport. Arxiv Link: https://arxiv.org/abs/2505.24759

## Overview

OT-MESH addresses the challenge of determining evolutionary relationships between cell types using single-cell RNA-seq data. By combining entropy-regularized optimal transport with the MESH refinement procedure, OT-MESH produces sparse and biologically interpretable correspondence matrices that reveal evolutionarily related cell types.

## Getting Started

### Prerequisites
- Python 3.10.12
- PyTorch
- NumPy
- Pandas
- Scanpy
- AnnData
- scikit-learn
- XGBoost
- Harmony-pytorch
- matplotlib
- seaborn
- refcm
- networkx


### Installation
Clone this repository and navigate to the project directory. Install dependencies with your preferred package manager (conda/pip). Some baselines (Harmony, RefCM, XGBoost) require additional packages as noted above.

## Quick Start

- Core usage (compute sparse OT with MESH):
  - Provide a pairwise distance matrix between source and target cell types.
  - Call `compute_sparse_transport` to obtain a sparse correspondence matrix.

Example:

```python
import pandas as pd
from compute_transport import compute_sparse_transport

# distance_matrix: DataFrame with source types as rows and target types as columns
W = compute_sparse_transport(
    distance_matrix,
    mesh_lr=1.0,       # MESH learning rate (λ)
    n_mesh_iters=4,    # MESH iterations (T)
    temperature=1.0,   # entropic regularization (α)
    n_sh_iters=5       # Sinkhorn iterations
)
```

## Methods

- OT-MESH core (`compute_transport.py`): entropy-regularized OT + MESH refinement
- Baselines:
  - Harmony+1NN (`compute_harmony.py`): integration followed by 1-NN assignment
  - XGBoost (`compute_xgboost.py`): reference-based classification
  - RefCM (external): used in benchmarking via the `refcm` package
- Evaluation (`evaluation_metrics.py`): sparseness, entropy, ARI, alignment scores
- Simulation (`simulate_data.py`): synthetic matched “species” datasets for benchmarking

## Repository Structure
```
├── Data_Preprocessing/
│   ├── data_preprocessing_mk_bc_per_fov.ipynb
│   ├── data_preprocessing_mk_ms_bc.ipynb
│   └── data_preprocessing_mk_ms_rgc.ipynb
├── Parameter_Selection/
│   ├── elbow_curves_alpha_*.png
│   ├── optimal_parameters.pkl
│   ├── optimal_parameters.json
│   └── transport_cost_selection.png
├── Scalability/
│   ├── scalability_benchmark_results_with_params.csv
│   ├── benchmark_parameters.csv
│   └── scalability_plots.png
├── Noise_Robustness/
│   ├── noise_robustness_results.csv
│   └── noise_robustness_plots.png
├── Plots/
│   ├── ot_bcs.png, ot_rgcs.png, ot_validation.png
│   ├── xgboost_confusion_matrix.png, harmony_1nn_confusion_matrix.png
│   ├── bc_correspondence_network.png, rgc_correspondence_network.png
│   └── refcm_transport_matrix.png
├── compute_transport.py
├── compute_harmony.py
├── compute_xgboost.py
├── evaluation_metrics.py
├── simulate_data.py
├── benchmark_scalability.py
├── benchmark_scalability.ipynb
├── benchmark_noise_robustness.ipynb
├── mk_bc_per_fov*.ipynb
├── mk_ms_bc*.ipynb
├── mk_ms_rgc*.ipynb
├── LICENSE.txt
└── README.md
```

## Benchmarks

- Scalability (`benchmark_scalability.py`, `benchmark_scalability.ipynb`):
  - Generates distance matrices from simulated data, sweeps OT-MESH parameters (α, λ, T), and records metrics.
  - Outputs results and plots to `Scalability/` and parameter selection artifacts to `Parameter_Selection/`.
- Noise Robustness (`benchmark_noise_robustness.ipynb`):
  - Evaluates stability of correspondences across noise levels; outputs to `Noise_Robustness/`.

## Parameter Selection

- Uses entropy elbow curves across λ (MESH LR) and T (MESH iterations) at fixed α.
- Given selected λ and T for each α, select α with lowest transport cost.
- Example artifacts: `Parameter_Selection/elbow_curves_alpha_*.png`, `optimal_parameters.{pkl,json}`, `transport_cost_selection.png`.

## Analysis Notebooks

- Validation (macaque peripheral vs foveal BC): `mk_bc_per_fov*.ipynb`
- Cross-species BC (mouse ↔ macaque): `mk_ms_bc*.ipynb`
- Cross-species RGC (mouse ↔ macaque): `mk_ms_rgc*.ipynb`
- Baseline comparisons: `mk_bc_per_fov_xgboost.ipynb`, `mk_bc_per_fov_harmony.ipynb`, `mk_bc_per_fov_refcm.ipynb`

## Data

- Shekhar et al. (2016): https://singlecell.broadinstitute.org/single_cell/study/SCP3/retinal-bipolar-neuron-drop-seq
- Tran et al. (2019): https://singlecell.broadinstitute.org/single_cell/study/SCP509/mouse-retinal-ganglion-cell-adult-atlas-and-optic-nerve-crush-time-series
- Peng et al. (2019): https://singlecell.broadinstitute.org/single_cell/study/SCP212/molecular-specification-of-retinal-cell-types-underlying-central-and-peripheral-vision-in-primates

Follow procedures from https://github.com/shekharlab/RetinaEvolution for preparing h5ad-formatted data.

## Citation

```bibtex
@misc{qiao2025unsupervisedevolutionarycelltype,
      title={Unsupervised Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport}, 
      author={Qiao, Mu},
      year={2025},
      eprint={2505.24759},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2505.24759}, 
}
```

## References

1. Peng et al. (2019). Molecular Classification and Comparative Taxonomics of Foveal and Peripheral Cells in Primate Retina. Cell.
2. Shekhar et al. (2016). Comprehensive Classification of Retinal Bipolar Neurons by Single-Cell Transcriptomics. Cell.
3. Tran et al. (2019). Single-cell profiles of retinal neurons differing in resilience to injury reveal neuroprotective genes. Neuron.
4. Hahn, J., Monavarfeshani, A., Qiao, M., et al. (2023). Evolution of neuronal cell classes and types in the vertebrate retina. Nature.
5. Galanti V. et al. (2024). Automated Cell Type Annotation with Reference Cluster Mapping. Biorxiv.

## License

This project is distributed under the terms described in `LICENSE.txt`.

## Contact

For questions or issues, please contact: muqiao0626@gmail.com
