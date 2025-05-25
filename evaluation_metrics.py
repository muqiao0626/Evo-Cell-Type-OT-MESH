import numpy as np
from sklearn.metrics import adjusted_rand_score
import time
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd

def sparseness_score(matrix: np.ndarray, threshold: float = 0.001) -> float:
    """
    Calculate the sparseness score of a correspondence matrix.
    
    Args:
        matrix: The correspondence matrix to evaluate
        threshold: Values below this threshold are considered "sparse" (default: 0.001)
        
    Returns:
        The fraction of entries in the matrix that are below the threshold
    """
    return np.mean(matrix < threshold)

def entropy(matrix: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the entropy of a correspondence matrix.
    
    Args:
        matrix: The correspondence matrix to evaluate
        epsilon: Small value to avoid log(0) (default: 1e-10)
        
    Returns:
        The entropy of the matrix
    """
    # Ensure matrix is normalized
    if not np.isclose(np.sum(matrix), 1.0):
        matrix = matrix / (np.sum(matrix) + epsilon)
    
    # Calculate entropy: -sum(p * log(p))
    return -np.sum(matrix * np.log(matrix + epsilon))

import numpy as np
from sklearn.metrics import adjusted_rand_score

def adjusted_rand_index_from_confusion_matrix(confusion_matrix: np.ndarray, sample_size: int = 10000):
    """
    Calculate Adjusted Rand Index from a confusion matrix containing proportions.
    
    Args:
        confusion_matrix: A matrix where cell [i,j] contains the proportion of 
                          instances with true label i and predicted label j.
        sample_size: The size of the synthetic dataset to generate for calculation.
                     Larger values give more precise results.
    
    Returns:
        The ARI score, ranging from -1 to 1
    """
    # Ensure the matrix is normalized
    if not np.isclose(confusion_matrix.sum(), 1.0):
        confusion_matrix = confusion_matrix / confusion_matrix.sum()
    
    # Create synthetic dataset from proportions
    true_labels = []
    pred_labels = []
    
    # For each cell in the matrix, generate appropriate number of samples
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            # Calculate how many instances to generate for this cell
            count = int(confusion_matrix[i, j] * sample_size)
            if count > 0:
                # Add these instances to our synthetic dataset
                true_labels.extend([i] * count)
                pred_labels.extend([j] * count)
    
    # Convert to arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Calculate ARI
    return adjusted_rand_score(true_labels, pred_labels)

def calculate_alignment_scores(W: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Calculate alignment scores for each cell pair (i,j) in the correspondence matrix W.
    
    The alignment score is defined as:
    s_ij = 1/2 * (W_ij/sum_i(W_ij) + W_ij/sum_j(W_ij))
    
    Args:
        W: The correspondence matrix where W[i,j] represents the connection strength
           between cell types i and j. Can be either numpy array or pandas DataFrame
        
    Returns:
        A matrix of the same shape as W containing the alignment scores
    """
    # Convert to numpy array if DataFrame
    if hasattr(W, 'values'):
        W = W.values
    
    # Avoid division by zero
    epsilon = 1e-10
    
    # Calculate row and column sums
    row_sums = np.sum(W, axis=1, keepdims=True) + epsilon
    col_sums = np.sum(W, axis=0, keepdims=True) + epsilon
    
    # Calculate normalized contributions
    row_contribution = W / col_sums  # Normalize by column sums
    col_contribution = W / row_sums  # Normalize by row sums
    
    # Calculate final alignment scores
    alignment_scores = 0.5 * (row_contribution + col_contribution)
    
    return alignment_scores 