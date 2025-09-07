import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from itertools import product
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Add RefCM import
from refcm import RefCM
from refcm.embeddings import HVGEmbedder
import anndata

# Import your existing implementations
from compute_transport import compute_sparse_transport, compute_regular_transport
from compute_harmony import harmony_1nn_mapping
from compute_xgboost import transcriptome_mapping
from evaluation_metrics import sparseness_score, entropy, adjusted_rand_index_from_confusion_matrix
from simulate_data import create_matched_species


def prepare_simulated_data_for_refcm(df, cluster_key='cell_type'):
    """
    Convert simulated DataFrame to AnnData format for RefCM.
    """
    # Create AnnData object
    adata = anndata.AnnData(X=df.values.astype(np.float32))
    adata.obs_names = [f'cell_{i}' for i in range(df.shape[0])]
    adata.var_names = df.columns
    
    # Add cluster information (cell types)
    adata.obs[cluster_key] = df.index.values
    
    # RefCM expects raw counts, ensure non-negative
    adata.X = np.maximum(adata.X, 0)
    
    return adata


def compute_transport_cost_only(transport_matrix, distance_matrix) -> float:
    """
    Compute only the transport cost (without entropy regularization).
    """
    if isinstance(transport_matrix, pd.DataFrame):
        W = transport_matrix.values
    elif torch.is_tensor(transport_matrix):
        W = transport_matrix.numpy() if transport_matrix.device.type == 'cpu' else transport_matrix.cpu().numpy()
    else:
        W = transport_matrix
    
    if isinstance(distance_matrix, pd.DataFrame):
        C = distance_matrix.values
    elif torch.is_tensor(distance_matrix):
        C = distance_matrix.numpy() if distance_matrix.device.type == 'cpu' else distance_matrix.cpu().numpy()
    else:
        C = distance_matrix
    
    return float(np.sum(W * C))


def run_parameter_sweep_with_entropy(
    distance_matrix,
    alpha_values: List[float],
    mesh_lr_values: List[float],
    mesh_iter_values: List[int],
    n_sh_iters: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform parameter sweep computing transport cost, sparsity, and entropy.
    """
    results = []
    combinations = list(product(alpha_values, mesh_lr_values, mesh_iter_values))
    total_combinations = len(combinations)
    
    if verbose:
        print(f"Starting parameter sweep with {total_combinations} combinations...")
        start_time = time.time()
    
    for i, (alpha, mesh_lr, mesh_iters) in enumerate(combinations):
        try:
            # Compute transport matrix
            transport_matrix = compute_sparse_transport(
                distance_matrix=distance_matrix,
                mesh_lr=mesh_lr,
                n_mesh_iters=mesh_iters,
                temperature=alpha,
                n_sh_iters=n_sh_iters
            )
            
            # Convert to numpy if needed
            W = transport_matrix.values if isinstance(transport_matrix, pd.DataFrame) else transport_matrix
            if torch.is_tensor(W):
                W = W.numpy() if W.device.type == 'cpu' else W.cpu().numpy()
            
            # Calculate metrics
            transport_cost = compute_transport_cost_only(transport_matrix, distance_matrix)
            sparsity = sparseness_score(W)
            entropy_val = entropy(W)
            
            results.append({
                'alpha': alpha,
                'mesh_lr': mesh_lr,
                'mesh_iters': mesh_iters,
                'transport_cost': transport_cost,
                'sparsity': float(sparsity),
                'entropy': float(entropy_val)
            })
            
        except Exception as e:
            if verbose:
                print(f"Failed for α={alpha}, λ={mesh_lr}, T={mesh_iters}: {e}")
            results.append({
                'alpha': alpha,
                'mesh_lr': mesh_lr,
                'mesh_iters': mesh_iters,
                'transport_cost': np.nan,
                'sparsity': np.nan,
                'entropy': np.nan
            })
        
        if verbose and (i+1) % max(1, total_combinations//10) == 0:
            elapsed = time.time() - start_time
            progress = (i+1) / total_combinations
            print(f"Progress: {i+1}/{total_combinations} ({progress:.1%})")
    
    if verbose:
        print(f"Parameter sweep completed in {time.time() - start_time:.1f} seconds!")
    
    return pd.DataFrame(results)


def plot_elbow_curves(results_df: pd.DataFrame, alpha: float, n_types: int,
                     output_dir: Optional[str] = None) -> Dict:
    """
    Plot entropy elbow curves for parameter selection.
    """
    alpha_df = results_df[results_df['alpha'] == alpha]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Entropy vs Lambda
    for iters in sorted(alpha_df['mesh_iters'].unique()):
        subset = alpha_df[alpha_df['mesh_iters'] == iters].sort_values('mesh_lr')
        ax1.plot(subset['mesh_lr'], subset['entropy'], 'o-', 
                label=f'T={iters}', linewidth=2, markersize=8)
    
    ax1.set_xlabel('MESH Learning Rate (λ)', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_title(f'Entropy vs λ (α={alpha}, n={n_types})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Entropy vs T
    for lr in sorted(alpha_df['mesh_lr'].unique()):
        subset = alpha_df[alpha_df['mesh_lr'] == lr].sort_values('mesh_iters')
        ax2.plot(subset['mesh_iters'], subset['entropy'], 's-', 
                label=f'λ={lr}', linewidth=2, markersize=8)
    
    ax2.set_xlabel('MESH Iterations (T)', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title(f'Entropy vs T (α={alpha}, n={n_types})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{n_types} Cell Types', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/elbow_curves_{n_types}types_alpha_{alpha}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'alpha': alpha, 'elbow_lambda': None, 'elbow_T': None, 'n_types': n_types}


def parameter_selection_workflow_for_scale(
    n_types: int,
    n_cells_per_type: int = 50,
    n_genes: int = 1000,
    output_dir: Optional[str] = "./Parameter_Selection"
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Complete parameter selection workflow for a specific scale.
    """
    print(f"\n{'='*60}")
    print(f"Parameter Selection for {n_types} Cell Types")
    print(f"{'='*60}")
    
    # Create dataset
    print("Creating dataset...")
    adata1, adata2, W_true = create_matched_species(
        n_types=n_types,
        n_genes=n_genes,
        n_cells_per_type=n_cells_per_type,
        scaling_method='uniform',
        scale_range=(0.8, 1.2),
        noise_level=1.0,
        gene_overlap_fraction=1.0,
        seed=43
    )
    
    # Normalize
    import scanpy as sc
    sc.pp.normalize_total(adata1, target_sum=1e4)
    sc.pp.log1p(adata1)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)
    
    # Create DataFrames
    df1 = pd.DataFrame(
        adata1.X,
        index=adata1.obs['cell_type_id'].values,
        columns=adata1.var_names
    )
    df2 = pd.DataFrame(
        adata2.X,
        index=adata2.obs['cell_type_id'].values,
        columns=adata2.var_names
    )
    
    # Select genes
    common_genes = list(set(df1.columns) & set(df2.columns))
    
    # Use appropriate number of genes based on scale
    if n_types <= 50:
        n_genes_use = min(50, len(common_genes))
    elif n_types <= 100:
        n_genes_use = min(100, len(common_genes))
    else:
        n_genes_use = min(200, len(common_genes))
    
    sc.pp.highly_variable_genes(adata1, n_top_genes=n_genes_use)
    hvg = adata1.var_names[adata1.var.highly_variable].tolist()
    selected_genes = [g for g in hvg if g in common_genes][:n_genes_use]
    
    print(f"Using {len(selected_genes)} genes")
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(df1, df2, selected_genes)
    if n_types < 50:
        alpha_values = [0.1, 0.5, 1.0, 5.0, 10.0]
        mesh_lr_values = [0.1, 0.5, 1.0, 5.0, 10.0]
        mesh_iter_values = [2, 4, 6, 8, 10]
    else:
        alpha_values = [0.1, 0.5, 1.0, 5.0, 10.0]
        mesh_lr_values = [0.5, 1.0, 5.0, 10.0, 50.0]
        mesh_iter_values = [8, 10, 12, 14, 16]
    
    # Run parameter sweep
    print(f"\nRunning parameter sweep...")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Lambda values: {mesh_lr_values}")
    print(f"  T values: {mesh_iter_values}")
    
    results_df = run_parameter_sweep_with_entropy(
        distance_matrix=distance_matrix,
        alpha_values=alpha_values,
        mesh_lr_values=mesh_lr_values,
        mesh_iter_values=mesh_iter_values,
        n_sh_iters=5,
        verbose=True
    )
    
    # Save results
    results_df.to_csv(f"{output_dir}/param_sweep_{n_types}types.csv", index=False)
    
    # Generate elbow curves for each alpha
    print(f"\nGenerating elbow curves for manual selection...")
    elbow_selections = []
    for alpha in sorted(results_df['alpha'].unique()):
        selection = plot_elbow_curves(results_df, alpha, n_types, output_dir)
        elbow_selections.append(selection)
    
    return elbow_selections, results_df, distance_matrix


def apply_elbow_selections(
    results_df: pd.DataFrame,
    elbow_selections: List[Dict],
    n_types: int,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Apply manual elbow selections to find optimal parameters.
    """
    valid_selections = []
    
    for selection in elbow_selections:
        if selection['elbow_lambda'] is not None and selection['elbow_T'] is not None:
            mask = ((results_df['alpha'] == selection['alpha']) & 
                   (results_df['mesh_lr'] == selection['elbow_lambda']) & 
                   (results_df['mesh_iters'] == selection['elbow_T']))
            
            if mask.sum() == 1:
                row = results_df[mask].iloc[0]
                valid_selections.append({
                    'alpha': selection['alpha'],
                    'mesh_lr': selection['elbow_lambda'],
                    'mesh_iters': selection['elbow_T'],
                    'transport_cost': row['transport_cost'],
                    'entropy': row['entropy'],
                    'sparsity': row['sparsity']
                })
    
    if not valid_selections:
        print("No valid selections found!")
        return None
    
    # Print transport costs for all alphas
    print("\n" + "="*60)
    print(f"TRANSPORT COSTS FOR EACH α ({n_types} types)")
    print("="*60)
    for sel in sorted(valid_selections, key=lambda x: x['alpha']):
        print(f"α = {sel['alpha']:6.3f}: Transport Cost = {sel['transport_cost']:.6f} "
              f"(λ={sel['mesh_lr']}, T={sel['mesh_iters']}, "
              f"Entropy={sel['entropy']:.4f}, Sparsity={sel['sparsity']:.3f})")
    
    # Select alpha with lowest transport cost
    best_params = min(valid_selections, 
                     key=lambda x: (x['transport_cost'], -x['alpha']))
    
    print("\n" + "="*60)
    print(f"OPTIMAL PARAMETERS FOR {n_types} TYPES")
    print("="*60)
    print(f"α = {best_params['alpha']}, λ = {best_params['mesh_lr']}, T = {best_params['mesh_iters']}")
    print(f"Transport Cost: {best_params['transport_cost']:.6f}")
    print(f"Entropy: {best_params['entropy']:.4f}")
    print(f"Sparsity: {best_params['sparsity']:.3f}")
    
    # Plot transport cost vs alpha
    if output_dir and len(valid_selections) > 1:
        plt.figure(figsize=(8, 6))
        
        data = pd.DataFrame(valid_selections)
        data = data.sort_values('alpha')
        
        plt.plot(data['alpha'], data['transport_cost'], 'o-', markersize=10, linewidth=2)
        # best_idx = data[data['alpha'] == best_params['alpha']].index[0]
        # plt.plot(data.loc[best_idx, 'alpha'], data.loc[best_idx, 'transport_cost'], 
        #         'r*', markersize=20, label='Selected')
        
        plt.xlabel('Regularization Parameter (α)', fontsize=12)
        plt.ylabel('Transport Cost', fontsize=12)
        plt.title(f'Transport Cost vs α at Elbow Points ({n_types} types)', fontsize=14)
        plt.xscale('log')
        # plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/transport_cost_selection_{n_types}types.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return best_params


def compute_distance_matrix(df1, df2, selected_genes=None):
    """Compute cosine distance matrix between cell type centroids."""
    if selected_genes is not None:
        df1 = df1[selected_genes]
        df2 = df2[selected_genes]
    
    type_means1 = df1.groupby(df1.index).mean()
    type_means2 = df2.groupby(df2.index).mean()
    
    tensor1 = torch.tensor(type_means1.values, dtype=torch.float32)
    tensor2 = torch.tensor(type_means2.values, dtype=torch.float32)
    
    norm1 = F.normalize(tensor1, dim=1)
    norm2 = F.normalize(tensor2, dim=1)
    
    distance_matrix = pd.DataFrame(
        1 - torch.mm(norm1, norm2.t()).numpy(),
        index=type_means1.index,
        columns=type_means2.index
    )
    
    return distance_matrix


def run_multi_scale_parameter_selection():
    """
    Run parameter selection workflow for multiple scales with manual elbow selection.
    """
    n_types_list = [20, 50, 100, 200, 500]
    output_dir = "./Parameter_Selection"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    all_selections = {}
    all_distance_matrices = {}
    
    # Step 1: Generate all parameter sweeps and plots
    for n_types in n_types_list:
        elbow_selections, results_df, distance_matrix = parameter_selection_workflow_for_scale(
            n_types=n_types,
            output_dir=output_dir
        )
        
        all_results[n_types] = results_df
        all_selections[n_types] = elbow_selections
        all_distance_matrices[n_types] = distance_matrix
    
    print("\n" + "="*80)
    print("MANUAL ELBOW SELECTION REQUIRED")
    print("="*80)
    print("Please review the generated plots and fill in the elbow points below.")
    print("Copy and modify the following code with your selections:\n")
    
    # Generate template code for manual selection
    for n_types in n_types_list:
        print(f"# {n_types} cell types")
        for i, selection in enumerate(all_selections[n_types]):
            alpha = selection['alpha']
            print(f"all_selections[{n_types}][{i}].update({{'elbow_lambda': None, 'elbow_T': None}})  # α={alpha}")
        print()
    
    return all_results, all_selections, all_distance_matrices


def apply_all_selections_and_benchmark(
    all_results: Dict,
    all_selections: Dict,
    all_distance_matrices: Dict,
    output_dir: str = "./Parameter_Selection"
):
    """
    Apply all manual selections and create final benchmark parameters.
    """
    optimal_params_all = {}
    
    for n_types in all_results.keys():
        print(f"\n{'='*60}")
        print(f"Processing {n_types} cell types")
        print(f"{'='*60}")
        
        optimal_params = apply_elbow_selections(
            all_results[n_types],
            all_selections[n_types],
            n_types,
            output_dir
        )
        
        if optimal_params:
            optimal_params['n_types'] = n_types
            optimal_params_all[n_types] = optimal_params
            
            # Test the selected parameters
            W_optimal = compute_sparse_transport(
                all_distance_matrices[n_types],
                mesh_lr=optimal_params['mesh_lr'],
                n_mesh_iters=int(optimal_params['mesh_iters']),
                temperature=optimal_params['alpha'],
                n_sh_iters=5
            )
            
            W = W_optimal.values if isinstance(W_optimal, pd.DataFrame) else W_optimal
            print(f"\nFinal verification:")
            print(f"  Entropy: {entropy(W):.4f}")
            print(f"  Sparsity: {sparseness_score(W):.3f}")
            print(f"  Transport Cost: {compute_transport_cost_only(W_optimal, all_distance_matrices[n_types]):.6f}")
    
    # Save all optimal parameters
    optimal_df = pd.DataFrame.from_dict(optimal_params_all, orient='index')
    optimal_df.to_csv(f"{output_dir}/optimal_params_manual_selection.csv")
    
    # Create summary plot
    if len(optimal_params_all) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        n_types_vals = sorted(optimal_params_all.keys())
        alphas = [optimal_params_all[n]['alpha'] for n in n_types_vals]
        lrs = [optimal_params_all[n]['mesh_lr'] for n in n_types_vals]
        iters = [optimal_params_all[n]['mesh_iters'] for n in n_types_vals]
        
        axes[0].loglog(n_types_vals, alphas, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Cell Types')
        axes[0].set_ylabel('Optimal α')
        axes[0].set_title('Temperature Scaling')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].semilogx(n_types_vals, lrs, 's-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Cell Types')
        axes[1].set_ylabel('Optimal λ')
        axes[1].set_title('Learning Rate Scaling')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].semilogx(n_types_vals, iters, '^-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Number of Cell Types')
        axes[2].set_ylabel('Optimal T')
        axes[2].set_title('Iterations Scaling')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Optimal Parameters Across Scales (Manual Selection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimal_params_scaling_manual.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return optimal_params_all

def compute_snr_genes(df, threshold=100):
    """
    Compute SNR for gene selection.
    """
    # Calculate counts, means, and variances per cell type
    type_counts = df.groupby(df.index).count()
    type_means = df.groupby(df.index).mean()
    type_vars = df.groupby(df.index).var()
    
    # Calculate SEM for each type
    type_sems = type_vars.div(type_counts - 1).pow(0.5)
    
    # Calculate overall mean expression for each gene
    overall_means = df.mean()
    
    # Calculate signal and noise
    signal = ((type_means - overall_means)**2).sum()
    noise = (type_sems**2).sum()
    
    # Calculate SNR
    snr = signal / np.maximum(noise, 1e-20)
    
    # Select high SNR genes
    high_snr_genes = snr[snr > threshold]
    
    return high_snr_genes.index.tolist()

def prepare_data_for_methods(adata1, adata2):
    """
    Prepare data in the format needed for each method.
    """
    # Convert cell type strings to numeric labels
    cell_types1 = adata1.obs['cell_type'].values
    cell_types2 = adata2.obs['cell_type'].values
    
    # Create numeric mapping
    unique_types1 = np.unique(cell_types1)
    unique_types2 = np.unique(cell_types2)
    
    # Create mapping dictionaries
    type_to_num1 = {f'Type_{i}': i for i in range(len(unique_types1))}
    type_to_num2 = {f'Type_{i}': i for i in range(len(unique_types2))}
    
    # Apply mapping
    numeric_labels1 = np.array([type_to_num1[ct] for ct in cell_types1])
    numeric_labels2 = np.array([type_to_num2[ct] for ct in cell_types2])
    
    # Create DataFrames with numeric indices
    df1 = pd.DataFrame(
        adata1.X,
        index=numeric_labels1,
        columns=adata1.var_names
    )
    df2 = pd.DataFrame(
        adata2.X,
        index=numeric_labels2,
        columns=adata2.var_names
    )
    
    return df1, df2, type_to_num1, type_to_num2


def compute_accuracy_metrics(W_pred, W_true):
    """
    Compute accuracy metrics for correspondence matrix.
    """
    # Ensure both matrices have the same shape by padding if necessary
    n_types1, n_types2 = W_true.shape
    
    # If W_pred is a DataFrame, convert to numpy
    if isinstance(W_pred, pd.DataFrame):
        W_pred = W_pred.values
    
    # Pad W_pred if necessary
    if W_pred.shape != W_true.shape:
        padded = np.zeros(W_true.shape)
        min_rows = min(W_pred.shape[0], W_true.shape[0])
        min_cols = min(W_pred.shape[1], W_true.shape[1])
        padded[:min_rows, :min_cols] = W_pred[:min_rows, :min_cols]
        W_pred = padded
    
    # Get predicted matches (highest value per row)
    pred_matches = np.argmax(W_pred, axis=1)
    true_matches = np.argmax(W_true, axis=1)
    
    # Only evaluate shared types
    n_shared = min(len(pred_matches), len(true_matches))
    pred_matches = pred_matches[:n_shared]
    true_matches = true_matches[:n_shared]
    
    accuracy = np.mean(pred_matches == true_matches)
    
    # Compute sparseness
    sparsity = sparseness_score(W_pred)
    
    # Compute entropy
    entropy_val = entropy(W_pred)
    
    # Compute ARI
    ari = adjusted_rand_index_from_confusion_matrix(W_pred)
    
    return {
        'accuracy': accuracy,
        'sparseness': sparsity,
        'entropy': entropy_val,
        'ari': ari
    }