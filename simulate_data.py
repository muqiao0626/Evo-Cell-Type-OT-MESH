import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import scanpy as sc
import anndata

def generate_cell_types(
    n_types: int = 100,
    n_genes: int = 500,
    n_cells_per_type: int = 50,
    noise_level: float = 1.0,
    seed: int = 43
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic cell types with housekeeping and marker genes.
    
    Parameters:
    -----------
    n_types : int
        Number of cell types to generate
    n_genes : int
        Total number of genes
    n_cells_per_type : int
        Number of cells per type
    noise_level : float
        Amount of noise to add
    seed : int
        Random seed
        
    Returns:
    --------
    expression_matrix : np.ndarray
        Cell x Gene expression matrix
    cell_labels : np.ndarray
        Cell type labels
    type_signatures : np.ndarray
        Mean expression for each cell type
    """
    np.random.seed(seed)
    
    # Divide genes into categories
    n_housekeeping = n_genes // 3  # Expressed in all cells
    n_marker_genes = n_genes - n_housekeeping  # Cell type specific genes
    markers_per_type = max(3, n_marker_genes // n_types)  # Each type gets some markers
    
    all_cells = []
    all_labels = []
    type_signatures = []
    
    for type_id in range(n_types):
        # Create signature for this cell type
        signature = np.zeros(n_genes)
        
        # Housekeeping genes - expressed in all cells with some variation
        signature[:n_housekeeping] = np.random.uniform(3, 7, n_housekeeping)
        
        # Marker genes - each cell type has unique markers
        # Distribute markers across types with some overlap for realism
        start_idx = n_housekeeping + (type_id * markers_per_type) % n_marker_genes
        end_idx = min(start_idx + markers_per_type, n_genes)
        
        # Primary markers - highly expressed
        signature[start_idx:end_idx] = np.random.uniform(10, 15, end_idx - start_idx)
        
        # Add some secondary markers with lower expression (creates more realistic patterns)
        n_secondary = min(5, n_marker_genes // 10)
        secondary_indices = np.random.choice(
            range(n_housekeeping, n_genes), 
            n_secondary, 
            replace=False
        )
        signature[secondary_indices] = np.random.uniform(5, 8, n_secondary)
        
        # Store the signature
        type_signatures.append(signature)
        
        # Generate cells with realistic noise
        # Use Poisson distribution for count data
        cells = np.random.poisson(
            signature[np.newaxis, :], 
            (n_cells_per_type, n_genes)
        )
        
        # Add Gaussian noise
        cells = cells + np.random.normal(0, noise_level, cells.shape)
        cells = np.maximum(cells, 0)
        
        all_cells.append(cells)
        all_labels.extend([type_id] * n_cells_per_type)
    
    expression_matrix = np.vstack(all_cells)
    cell_labels = np.array(all_labels)
    type_signatures = np.array(type_signatures)
    
    return expression_matrix, cell_labels, type_signatures


def create_matched_species(
    n_types: int = 100,
    n_genes: int = 500,
    n_cells_per_type: int = 50,
    scaling_method: str = 'uniform',
    scale_range: Tuple[float, float] = (0.8, 1.2),
    noise_level: float = 1.0,
    gene_overlap_fraction: float = 1.0,
    seed: int = 43
) -> Tuple[anndata.AnnData, anndata.AnnData, np.ndarray]:
    """
    Create matched realistic datasets for two "species".
    
    Parameters:
    -----------
    n_types : int
        Number of cell types
    n_genes : int
        Total number of genes
    n_cells_per_type : int
        Number of cells per type
    scaling_method : str
        How to scale expression between species ('uniform', 'random', 'gradual')
    scale_range : Tuple[float, float]
        Range for scaling factors
    noise_level : float
        Amount of noise to add
    gene_overlap_fraction : float
        Fraction of genes shared between species
    seed : int
        Random seed
        
    Returns:
    --------
    adata1, adata2 : anndata.AnnData
        Matched datasets for two "species"
    true_correspondence : np.ndarray
        Ground truth correspondence matrix
    """
    np.random.seed(seed)
    
    # Generate species 1
    expr1, labels1, signatures1 = generate_cell_types(
        n_types=n_types,
        n_genes=n_genes,
        n_cells_per_type=n_cells_per_type,
        noise_level=noise_level,
        seed=seed
    )
    
    # Generate species 2 with scaled signatures
    signatures2 = signatures1.copy()
    
    if scaling_method == 'uniform':
        # Each cell type scaled uniformly
        for i in range(n_types):
            scale = np.random.uniform(*scale_range)
            signatures2[i] *= scale
            
    elif scaling_method == 'random':
        # Each gene scaled independently
        for i in range(n_types):
            scales = np.random.uniform(*scale_range, size=n_genes)
            signatures2[i] *= scales
            
    elif scaling_method == 'gradual':
        # Scaling increases with type index (simulating evolutionary distance)
        for i in range(n_types):
            scale = scale_range[0] + (scale_range[1] - scale_range[0]) * (i / max(n_types - 1, 1))
            signatures2[i] *= scale
    
    # Generate species 2 cells from scaled signatures
    all_cells2 = []
    labels2 = []
    
    for type_id, signature in enumerate(signatures2):
        # Generate cells with Poisson noise
        cells = np.random.poisson(
            signature[np.newaxis, :], 
            (n_cells_per_type, n_genes)
        )
        cells = cells + np.random.normal(0, noise_level, cells.shape)
        cells = np.maximum(cells, 0)
        
        all_cells2.append(cells)
        labels2.extend([type_id] * n_cells_per_type)
    
    expr2 = np.vstack(all_cells2)
    labels2 = np.array(labels2)
    
    # Create gene names
    gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    # Handle gene overlap
    gene_names1 = gene_names.copy()
    gene_names2 = gene_names.copy()
    
    if gene_overlap_fraction < 1.0:
        n_overlap = int(n_genes * gene_overlap_fraction)
        overlap_indices = np.random.choice(n_genes, n_overlap, replace=False)
        
        for i in range(n_genes):
            if i not in overlap_indices:
                gene_names2[i] = f"Gene_{i}_sp2"
    
    # Create cell type names
    cell_type_names = [f'Type_{i}' for i in range(n_types)]
    
    # Create AnnData objects
    adata1 = anndata.AnnData(
        X=expr1,
        obs=pd.DataFrame({
            'cell_type': [cell_type_names[i] for i in labels1],
            'cell_type_id': labels1
        }),
        var=pd.DataFrame(index=gene_names1)
    )
    
    adata2 = anndata.AnnData(
        X=expr2,
        obs=pd.DataFrame({
            'cell_type': [cell_type_names[i] for i in labels2],
            'cell_type_id': labels2
        }),
        var=pd.DataFrame(index=gene_names2)
    )
    
    # Create ground truth correspondence matrix (identity for matched types)
    true_correspondence = np.eye(n_types)

    if not np.isclose(true_correspondence.sum(), 1.0):
        true_correspondence = true_correspondence / true_correspondence.sum()
    
    # Store metadata
    adata1.uns['signatures'] = signatures1
    adata1.uns['n_types'] = n_types
    
    adata2.uns['signatures'] = signatures2
    adata2.uns['n_types'] = n_types
    
    return adata1, adata2, true_correspondence