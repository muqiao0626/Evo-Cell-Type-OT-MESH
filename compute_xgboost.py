import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def transcriptome_mapping(mk_data, ms_data, labels_mk, labels_ms=None, mk_species="Species1", ms_species="Species2", p_cutoff=0, max_cells_per_type=400, train_frac=0.6, n_round = 1, seed=1000, xgb_params=None):
    """
    Cross-species transcriptome mapping using XGBoost.
    
    Parameters:
    -----------
    mk_data : DataFrame
        Expression matrix for species 1 training species
    ms_data : DataFrame
        Expression matrix for species 2 test species  
    labels_mk : Series or array
        Cell type labels for species 1
    labels_ms : Series or array, optional
        Cell type labels for species 2 (for evaluation, not used in prediction)
    mk_species : str
        Name of species 1
    ms_species : str
        Name of species 2
    n_features : int
        Number of features to use
    p_cutoff : float
        Probability cutoff for predictions
    max_cells_per_type : int
        Maximum cells per cell type for training
    train_frac : float
        Fraction of cells to use for training
    n_round: int
        Number of rounds to train the model
    seed : int
        Random seed
    xgb_params : dict
        Additional parameters for XGBClassifier
        
    Returns:
    --------
    confusion_matrix : DataFrame
        Confusion matrix of predicted vs true cell types
    model : XGBClassifier
        Trained XGBoost model
    """
    np.random.seed(seed)
    
    print("Processing input data...")
    
    # Select common genes between the two species (simulating ortholog finding)
    common_genes = np.intersect1d(mk_data.columns, ms_data.columns)
    print(f"Using {len(common_genes)} common genes between species")
    
    # Subset to common genes
    mk_data = mk_data[common_genes]
    ms_data = ms_data[common_genes]
    
    # Scale data
    scaler = StandardScaler()
    mk_data_scaled = pd.DataFrame(
        scaler.fit_transform(mk_data), 
        index=mk_data.index, 
        columns=mk_data.columns
    )
    ms_data_scaled = pd.DataFrame(
        scaler.transform(ms_data),
        index=ms_data.index,
        columns=ms_data.columns
    )
    
    # Training and validation splitting
    unique_labels = np.unique(labels_mk)
    train_indices = []
    val_indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels_mk == label)[0]
        n_cells = min(max_cells_per_type, int(len(label_indices) * train_frac))
        
        if n_cells == 0:
            continue
            
        selected_indices = np.random.choice(label_indices, n_cells, replace=False)
        train_indices.extend(selected_indices)
        
        remaining_indices = np.setdiff1d(label_indices, selected_indices)
        val_indices.extend(remaining_indices)
    
    X_train = mk_data_scaled.iloc[train_indices]
    y_train = [labels_mk[i] for i in train_indices]
    
    X_val = mk_data_scaled.iloc[val_indices]
    y_val = [labels_mk[i] for i in val_indices]
    
    print("Training XGBoost model...")
    
    # Set default parameters for XGBClassifier if not provided
    if xgb_params is None:
        xgb_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'eta': 0.2,
            'max_depth': 6,
            'subsample': 0.6,
            'use_label_encoder': False,
            'early_stopping_rounds': 10,
            'random_state': seed
        }
    
    # XGBoost model training
    model = XGBClassifier(**xgb_params)

    # Initial fit without xgb_model parameter
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Continue training for additional rounds if needed
    if n_round > 1:
        for _ in range(n_round - 1):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                xgb_model=model.get_booster()  # Use the fitted booster for continued training
            )
    
    # Validate on hold-out set
    val_preds = model.predict(X_val)
    val_accuracy = np.mean(val_preds == y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Predict on second species
    print(f"Predicting cell types for {ms_species}...")
    ms_probs = model.predict_proba(ms_data_scaled)
    
    # Get predictions and confidence
    ms_pred_indices = np.argmax(ms_probs, axis=1)
    ms_pred_labels = [model.classes_[i] for i in ms_pred_indices]
    ms_pred_confidence = np.max(ms_probs, axis=1)
    
    # Apply cutoff
    if p_cutoff > 0:
        ms_pred_labels = [
            label if conf >= p_cutoff else "Unknown" 
            for label, conf in zip(ms_pred_labels, ms_pred_confidence)
        ]
    
    # Create confusion matrix
    if labels_ms is not None:
        print("Creating confusion matrix...")
        cm = pd.crosstab(
            pd.Series(labels_ms, name='True'), 
            pd.Series(ms_pred_labels, name='Predicted'),
            normalize='index'
        )
        
        # Handle missing cell types in predictions
        for label in unique_labels:
            if label not in cm.columns:
                cm[label] = 0
                
        # Sort columns
        cm = cm.reindex(columns=unique_labels)
        
        # Row normalization
        cm = cm.div(cm.sum(axis=1), axis=0)
        
        return cm, model
    else:
        return pd.Series(ms_pred_labels, index=ms_data.index), model