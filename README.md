# OT-MESH: Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport

This repository contains the implementation of OT-MESH (Optimal Transport with Minimize Entropy of Sinkhorn) for identifying evolutionary correspondences between cell types across species. The method is described in:

Qiao, M. (2025) Unsupervised Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport. *Preprint*.

## Overview

OT-MESH addresses the challenge of systematically determining evolutionary relationships between cell types across species using single-cell RNA sequencing data. By combining entropy-regularized optimal transport with the MESH (Minimize Entropy of Sinkhorn) refinement procedure, this approach produces sparse and biologically interpretable correspondence matrices that reveal evolutionarily related cell types.

## Getting Started

### Prerequisites
- Python 3.10.12
- PyTorch
- NumPy
- Pandas
- Scanpy
- scikit-learn
- XGBoost
- Harmony-pytorch

### Installation
Clone this repository and navigate to the project directory

## Quick Start

The repo consists of the following core modules:

1. **Optimal Transport with MESH** (`compute_transport.py`): Core implementation of entropy-regularized OT with MESH refinement
2. **Baseline Methods**:
   - `compute_harmony.py`: Harmony integration with 1-NN matching
   - `compute_xgboost.py`: XGBoost reference-based classification
3. **Evaluation** (`evaluation_metrics.py`): Metrics for assessing correspondence quality

## Repository Structure
```
Evo_Cell_Type_OT_MESH/
├── Data_Preprocessing/
│   ├── data_preprocessing_mk_bc_per_fov.ipynb  # Macaque BC peripheral/foveal preprocessing
│   ├── data_preprocessing_mk_ms_bc.ipynb       # Mouse-macaque BC preprocessing
│   └── data_preprocessing_mk_ms_rgc.ipynb      # Mouse-macaque RGC preprocessing
├── compute_transport.py    # Core OT-MESH implementation
├── compute_harmony.py      # Harmony baseline method
├── compute_xgboost.py      # XGBoost baseline method
├── evaluation_metrics.py   # Evaluation metrics (sparseness, entropy, ARI)
├── mk_bc_per_fov*.ipynb   # Macaque peripheral-foveal BC analyses
├── mk_ms_bc*.ipynb        # Mouse-macaque BC analyses
├── mk_ms_rgc*.ipynb       # Mouse-macaque RGC analyses
└── README.md
```

## Analysis Notebooks

### Validation Experiments
- `mk_bc_per_fov.ipynb`: Main analysis for macaque peripheral vs. foveal BC correspondence
- `mk_bc_per_fov_parameter_selection.ipynb`: Parameter optimization via grid search
- `mk_bc_per_fov_xgboost.ipynb`: XGBoost baseline comparison
- `mk_bc_per_fov_harmony.ipynb`: Harmony+1NN baseline comparison

### Cross-Species Analyses
- `mk_ms_bc.ipynb`: Cross-species BC correspondence between mouse and macaque
- `mk_ms_bc_parameter_selection.ipynb`: Parameter selection for BC analysis
- `mk_ms_rgc.ipynb`: Cross-species RGC correspondence between mouse and macaque
- `mk_ms_rgc_parameter_selection.ipynb`: Parameter selection for RGC analysis

## Data
- Shekhar et al. (2016): https://singlecell.broadinstitute.org/single_cell/study/SCP3/retinal-bipolar-neuron-drop-seq
- Tran et al. (2019): https://singlecell.broadinstitute.org/single_cell/study/SCP509/mouse-retinal-ganglion-cell-adult-atlas-and-optic-nerve-crush-time-series
- Peng et al. (2019): https://singlecell.broadinstitute.org/single_cell/study/SCP212/molecular-specification-of-retinal-cell-types-underlying-central-and-peripheral-vision-in-primates

Follow procedures from this repo (https://github.com/shekharlab/RetinaEvolution) for the h5ad data format.

## Citation
```bibtex
@article{qiao2025otmesh,
  title={Unsupervised Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport},
  author={Qiao, Mu},
  journal={Preprint},
  year={2025}
}
```

## References

Key datasets used in this study:
1. Peng et al. (2019). Molecular Classification and Comparative Taxonomics of Foveal and Peripheral Cells in Primate Retina. *Cell*.
2. Shekhar et al. (2016). Comprehensive Classification of Retinal Bipolar Neurons by Single-Cell Transcriptomics. *Cell*.
3. Tran et al. (2019). Single-cell profiles of retinal neurons differing in resilience to injury reveal neuroprotective genes. *Neuron*.

## Contact

For questions or issues, please contact: muqiao0626@gmail.com