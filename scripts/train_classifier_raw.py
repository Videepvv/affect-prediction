"""
Multi-label classification for affect prediction using:
- BERT embeddings for transcripts
- Prosodic features
- Combined features

Models: AdaBoost, Random Forest, SVM
Evaluation: AUROC per label with Leave-One-Group-Out cross-validation
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer, BertModel
import torch

warnings.filterwarnings('ignore')


def get_bert_embeddings(texts: List[str], model_name: str = 'bert-base-uncased', 
                        batch_size: int = 32) -> np.ndarray:
    """
    Extract BERT embeddings for a list of texts.
    Uses [CLS] token embedding as sentence representation.
    
    Args:
        texts: List of text strings
        model_name: BERT model name
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape (n_samples, 768)
    """
    print(f"Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    embeddings = []
    
    print(f"Extracting BERT embeddings for {len(texts)} texts...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Handle None/NaN values
        batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
        
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return np.vstack(embeddings)


def get_prosodic_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract prosodic features from the dataframe.
    
    Returns:
        Tuple of (feature array, feature names)
    """
    # Prosodic feature columns (based on openSMILE eGeMAPS features)
    prosodic_cols = [col for col in df.columns if any(
        keyword in col for keyword in [
            'F0', 'loudness', 'spectral', 'mfcc', 'jitter', 'shimmer',
            'HNR', 'logRel', 'frequency', 'bandwidth', 'amplitude',
            'alpha', 'hammarberg', 'slope', 'Voiced', 'Unvoiced',
            'equivalentSoundLevel', 'Peaks'
        ]
    )]
    
    print(f"Found {len(prosodic_cols)} prosodic feature columns")
    
    features = df[prosodic_cols].values
    
    # Handle NaN values - replace with column mean
    col_means = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        mask = np.isnan(features[:, i])
        features[mask, i] = col_means[i]
    
    return features, prosodic_cols


def prepare_labels(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], MultiLabelBinarizer]:
    """
    Prepare multi-label targets from label1, label2, label3 columns.
    
    Returns:
        Tuple of (binary label matrix, label classes, binarizer)
    """
    label_cols = ['label1', 'label2', 'label3']
    
    # Collect all labels for each sample
    labels_list = []
    for _, row in df.iterrows():
        sample_labels = []
        for col in label_cols:
            if pd.notna(row[col]) and row[col] != '':
                sample_labels.append(row[col])
        labels_list.append(sample_labels)
    
    # Find samples with at least one label
    valid_mask = [len(labels) > 0 for labels in labels_list]
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels_list)
    
    print(f"Label classes: {mlb.classes_}")
    print(f"Total samples: {len(labels_list)}")
    print(f"Samples with labels: {sum(valid_mask)}")
    
    # Label distribution
    print("\nLabel distribution:")
    for i, label in enumerate(mlb.classes_):
        count = y[:, i].sum()
        print(f"  {label}: {count} ({100*count/len(y):.1f}%)")
    
    return y, list(mlb.classes_), mlb, np.array(valid_mask)


def get_models() -> Dict[str, OneVsRestClassifier]:
    """
    Get dictionary of multi-label classifiers.
    """
    models = {
        'RandomForest': OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        'AdaBoost': OneVsRestClassifier(
            AdaBoostClassifier(n_estimators=100, random_state=42)
        ),
        'SVM': OneVsRestClassifier(
            SVC(kernel='rbf', probability=True, random_state=42)
        )
    }
    return models


def evaluate_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  model_name: str, model: OneVsRestClassifier,
                  label_classes: List[str]) -> Dict[str, float]:
    """
    Evaluate model using Leave-One-Group-Out cross-validation.
    
    Returns:
        Dictionary of AUROC scores per label and macro average
    """
    logo = LeaveOneGroupOut()
    
    # Store predictions and true labels for each fold
    all_y_true = []
    all_y_pred_proba = []
    
    n_splits = logo.get_n_splits(X, y, groups)
    print(f"  Running {n_splits}-fold LOGO cross-validation...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        all_y_true.append(y_test)
        all_y_pred_proba.append(y_pred_proba)
    
    # Concatenate all folds
    y_true_all = np.vstack(all_y_true)
    y_pred_proba_all = np.vstack(all_y_pred_proba)
    
    # Calculate AUROC for each label
    auroc_scores = {}
    valid_aurocs = []
    
    for i, label in enumerate(label_classes):
        y_true_label = y_true_all[:, i]
        y_pred_label = y_pred_proba_all[:, i]
        
        # Check if both classes are present
        if len(np.unique(y_true_label)) > 1:
            auroc = roc_auc_score(y_true_label, y_pred_label)
            auroc_scores[label] = auroc
            valid_aurocs.append(auroc)
        else:
            auroc_scores[label] = np.nan
            print(f"    Warning: Label '{label}' has only one class in test set")
    
    # Macro average (only over valid labels)
    auroc_scores['macro_avg'] = np.mean(valid_aurocs) if valid_aurocs else np.nan
    
    return auroc_scores


def run_experiments(df: pd.DataFrame, feature_type: str = 'all',
                    bert_model: str = 'bert-base-uncased') -> pd.DataFrame:
    """
    Run multi-label classification experiments.
    
    Args:
        df: Preprocessed dataframe
        feature_type: 'prosodic', 'transcript', or 'all'
        bert_model: BERT model name for transcript features
    
    Returns:
        DataFrame with results
    """
    print("=" * 60)
    print(f"Running experiments with feature type: {feature_type}")
    print("=" * 60)
    
    # Prepare labels
    y, label_classes, mlb, valid_mask = prepare_labels(df)
    
    # Filter to samples with valid labels
    df_valid = df[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    
    # Get groups for LOGO
    groups = df_valid['groupID'].values
    print(f"\nUnique groups: {np.unique(groups)}")
    
    # Prepare features based on type
    features_dict = {}
    
    if feature_type in ['prosodic', 'all']:
        prosodic_features, prosodic_cols = get_prosodic_features(df_valid)
        features_dict['prosodic'] = prosodic_features
        print(f"Prosodic features shape: {prosodic_features.shape}")
    
    if feature_type in ['transcript', 'all']:
        bert_embeddings = get_bert_embeddings(
            df_valid['transcript'].tolist(),
            model_name=bert_model
        )
        features_dict['transcript'] = bert_embeddings
        print(f"BERT embeddings shape: {bert_embeddings.shape}")
    
    if feature_type == 'all':
        combined_features = np.hstack([features_dict['prosodic'], features_dict['transcript']])
        features_dict['combined'] = combined_features
        print(f"Combined features shape: {combined_features.shape}")
    
    # Get models
    models = get_models()
    
    # Results storage
    results = []
    
    # Run experiments for each feature set
    feature_sets = [feature_type] if feature_type in ['prosodic', 'transcript'] else ['prosodic', 'transcript', 'combined']
    
    for feat_name in feature_sets:
        X = features_dict[feat_name]
        
        print(f"\n{'='*40}")
        print(f"Feature set: {feat_name}")
        print(f"{'='*40}")
        
        for model_name, model in models.items():
            print(f"\nModel: {model_name}")
            
            # Create fresh model instance
            model = get_models()[model_name]
            
            auroc_scores = evaluate_logo(X, y, groups, model_name, model, label_classes)
            
            # Store results
            result = {
                'feature_type': feat_name,
                'model': model_name,
                'macro_auroc': auroc_scores['macro_avg']
            }
            for label in label_classes:
                result[f'auroc_{label}'] = auroc_scores.get(label, np.nan)
            
            results.append(result)
            
            # Print results
            print(f"    Macro AUROC: {auroc_scores['macro_avg']:.4f}")
            for label in label_classes:
                score = auroc_scores.get(label, np.nan)
                if not np.isnan(score):
                    print(f"    {label}: {score:.4f}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-label affect classification"
    )
    parser.add_argument(
        "input_file",
        help="Path to preprocessed CSV file"
    )
    parser.add_argument(
        "-f", "--features",
        choices=['prosodic', 'transcript', 'all'],
        default='all',
        help="Feature type to use (default: all)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for results CSV",
        default=None
    )
    parser.add_argument(
        "--bert-model",
        default='bert-base-uncased',
        help="BERT model to use (default: bert-base-uncased)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples")
    
    # Run experiments
    results_df = run_experiments(df, args.features, args.bert_model)
    
    # Save results
    if args.output is None:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"classification_results_{args.features}.csv"
    else:
        output_path = args.output
    
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nResults Summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
