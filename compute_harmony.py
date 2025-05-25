import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import scanpy as sc
import time
warnings.filterwarnings('ignore')

def construct_data(source_df, target_df, source_labels, target_labels, source_name="Peripheral", target_name="Fovea"):
    """
    Construct AnnData objects from source and target dataframes and concatenate them.
    
    Parameters:
    -----------
    source_df : DataFrame
        Source dataset
    target_df : DataFrame
        Target dataset
    source_labels : array-like
        Cell type labels for source
    target_labels : array-like
        Cell type labels for target
    source_name : str
        Name for source batch
    target_name : str
        Name for target batch
        
    Returns:
    --------
    adata : AnnData
        Concatenated AnnData object
    """
    # Create AnnData objects
    adata_source = sc.AnnData(X=source_df.values, obs=pd.DataFrame(index=[f"source_{i}" for i in range(len(source_df))]))
    adata_target = sc.AnnData(X=target_df.values, obs=pd.DataFrame(index=[f"target_{i}" for i in range(len(target_df))]))
    
    # Add cell type information
    adata_source.obs['cell_type'] = source_labels
    adata_target.obs['cell_type'] = target_labels
    
    # Add batch information
    adata_source.obs['batch'] = source_name
    adata_target.obs['batch'] = target_name
    
    # Ensure same gene order
    common_genes = list(source_df.columns)
    adata_source.var_names = common_genes
    adata_target.var_names = common_genes
    
    # Concatenate
    adata = adata_source.concatenate(adata_target, batch_key='dataset')
    
    return adata

def harmony_1nn_mapping(source_df, target_df, source_labels, target_labels, 
                        source_name="Peripheral", target_name="Fovea"):
    """
    Perform Harmony integration followed by 1NN assignment
    
    Parameters:
    -----------
    source_df : DataFrame
        Source dataset (peripheral)
    target_df : DataFrame
        Target dataset (foveal)
    source_labels : array-like
        Cell type labels for source
    target_labels : array-like
        Cell type labels for target
    source_name : str
        Name for source batch
    target_name : str
        Name for target batch
        
    Returns:
    --------
    confusion_matrix : DataFrame
        Confusion matrix of assignments
    adata_integrated : AnnData
        Integrated AnnData object
    """
    
    # Construct data
    adata = construct_data(source_df, target_df, source_labels, target_labels, 
                          source_name, target_name)
    
    # Start timing
    start_time = time.time()

    print("Running Harmony integration...")
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # Run PCA
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    
    # Run Harmony
    sc.external.pp.harmony_integrate(adata, 'batch', basis='X_pca', adjusted_basis='X_pca_harmony')
    
    # Extract integrated representations
    harmony_coords = adata.obsm['X_pca_harmony']
    
    # Split back into source and target using the batch information
    source_mask = adata.obs['batch'] == source_name
    target_mask = adata.obs['batch'] == target_name
    
    source_coords = harmony_coords[source_mask]
    target_coords = harmony_coords[target_mask]
    
    # Perform 1NN assignment
    print("Performing 1NN assignment...")
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(target_coords)
    
    # Find nearest neighbors for source cells
    distances, indices = nn.kneighbors(source_coords)
    
    # Get predicted labels
    predicted_labels = target_labels[indices.flatten()]
    
    # Create confusion matrix
    unique_source = np.unique(source_labels)
    unique_target = np.unique(target_labels)
    
    confusion_matrix = pd.DataFrame(
        np.zeros((len(unique_target), len(unique_source))),
        index=unique_target,
        columns=unique_source
    )
    
    # Fill confusion matrix
    for true_label, pred_label in zip(source_labels, predicted_labels):
        confusion_matrix.loc[pred_label, true_label] += 1
    
    # Normalize by column (each column sums to 1)
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=0), axis=1)

    # End timing
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running Time: {running_time:.6f} seconds")
    
    return confusion_matrix