"""
Multi-label Classification with BERT - Including "None" Emotion Class
Filters to 8 emotions: Optimistic, Curious, Confused, Conflicted, Surprised, Disengaged, Frustrated, None
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import sys
import os
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

def get_bert_embeddings(texts, model_name='bert-base-uncased', batch_size=16):
    """Extract BERT embeddings for a list of texts"""
    print(f"\n{'='*60}")
    print(f"Loading BERT model: {model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Device: {device}")
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    embeddings = []
    
    print(f"\nExtracting embeddings for {len(texts)} texts...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(batch_texts, 
                              padding=True, 
                              truncation=True, 
                              max_length=512, 
                              return_tensors='pt')
            
            # Move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get embeddings
            outputs = model(**encoded)
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings

def prepare_multilabel_data(file_path, file_type='csv'):
    """Prepare data with multi-label encoding including None class"""
    print(f"\nLoading data from: {file_path}")
    
    if file_type == 'excel':
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Original shape: {df.shape}")
    
    # Define the emotions we want to keep (7 emotions + None)
    target_emotions = ['Optimistic', 'Curious', 'Confused', 'Conflicted', 
                      'Surprised', 'Disengaged', 'Frustrated']
    all_labels = target_emotions + ['None']  # Add None as 8th class
    
    print(f"\nTarget emotions: {target_emotions}")
    print(f"All labels (including None): {all_labels}")
    
    # Get prosodic feature columns (eGeMAPS features)
    prosodic_cols = [col for col in df.columns if any(x in col for x in 
                    ['F0', 'jitter', 'shimmer', 'HNR', 'loudness', 'spectral', 
                     'mfcc', 'alpha', 'hammarberg', 'slope'])]
    
    print(f"Found {len(prosodic_cols)} prosodic features")
    
    # Remove rows with missing prosodic features
    df = df.dropna(subset=prosodic_cols)
    print(f"After removing rows with missing prosodic features: {df.shape}")
    
    # Find timestamp column (handle different names/encodings)
    timestamp_col = None
    for col in df.columns:
        if 'timestamp' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col is None:
        # Try using videoTime as fallback
        if 'videoTime' in df.columns:
            timestamp_col = 'videoTime'
        else:
            raise ValueError("Could not find timestamp column")
    
    print(f"Using timestamp column: {timestamp_col}")
    
    # Create unique identifier for each time window
    df['window_id'] = df[timestamp_col].astype(str) + '_' + df['participantID'].astype(str)
    
    # Filter and process labels
    filtered_rows = []
    
    for window_id, group in df.groupby('window_id'):
        # Get all unique labels for this window
        labels_list = []
        has_invalid_label = False
        
        for labels_str in group['labels'].dropna():
            if pd.notna(labels_str) and str(labels_str).strip() != '':
                emotions = [l.strip() for l in str(labels_str).split(',')]
                
                # Check if any emotion is not in our target list
                for emotion in emotions:
                    if emotion not in target_emotions:
                        has_invalid_label = True
                        break
                
                if not has_invalid_label:
                    labels_list.extend(emotions)
        
        # Skip windows with invalid labels
        if has_invalid_label:
            continue
        
        # Get unique labels
        unique_labels = list(set(labels_list))
        
        # If no labels at all, this is "None"
        if len(unique_labels) == 0:
            unique_labels = ['None']
        
        # Create multi-label encoding
        result = {label: 1 if label in unique_labels else 0 for label in all_labels}
        
        # Add other columns (take first value since they should be the same)
        result['groupID'] = group['groupID'].iloc[0]
        result['participantID'] = group['participantID'].iloc[0]
        result['transcript'] = group['transcript'].iloc[0] if 'transcript' in group.columns else ''
        
        # Add prosodic features (average if multiple values)
        for col in prosodic_cols:
            result[col] = group[col].mean()
        
        filtered_rows.append(result)
    
    df_agg = pd.DataFrame(filtered_rows)
    
    print(f"\nAfter aggregating and filtering: {df_agg.shape[0]} unique windows")
    
    # Count windows with transcripts
    has_transcript = df_agg['transcript'].notna() & (df_agg['transcript'] != '') & (df_agg['transcript'] != 'nan')
    print(f"Windows with transcripts: {has_transcript.sum()} ({has_transcript.sum()/len(df_agg)*100:.1f}%)")
    
    # Filter for windows with valid transcripts
    df_agg = df_agg[has_transcript].copy()
    print(f"After filtering for valid transcripts: {len(df_agg)} windows")
    
    # Check for windows with multiple labels
    label_counts = df_agg[all_labels].sum(axis=1)
    print(f"\nLabel distribution per window:")
    print(f"  1 label: {(label_counts == 1).sum()} windows")
    print(f"  2 labels: {(label_counts == 2).sum()} windows")
    print(f"  3+ labels: {(label_counts >= 3).sum()} windows")
    print(f"  Max labels in one window: {label_counts.max()}")
    
    # Label frequencies
    print(f"\nLabel frequencies:")
    for label in all_labels:
        count = df_agg[label].sum()
        print(f"  {label:20s}: {count} ({count/len(df_agg)*100:.1f}%)")
    
    X_prosody = df_agg[prosodic_cols].values
    y = df_agg[all_labels].values
    groups = df_agg['groupID'].values
    transcripts = df_agg['transcript'].values
    
    # Clean transcripts - replace any remaining NaN/None with empty string
    transcripts = [str(t) if pd.notna(t) and str(t) not in ['nan', 'None', ''] else 'N/A' for t in transcripts]
    
    print(f"\nFinal dataset:")
    print(f"  X_prosody shape: {X_prosody.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Number of groups: {len(np.unique(groups))}")
    print(f"  Groups: {sorted(np.unique(groups))}")
    
    return X_prosody, y, groups, transcripts, all_labels

def prepare_features(X_prosody, transcripts, groups, modality='combined'):
    """Prepare features based on modality"""
    
    if modality == 'prosody':
        print(f"\n{'='*60}")
        print("Using PROSODY ONLY features")
        print(f"{'='*60}")
        return X_prosody, groups
    
    elif modality == 'text':
        print(f"\n{'='*60}")
        print("Using TEXT (BERT) ONLY features")
        print(f"{'='*60}")
        
        X_bert = get_bert_embeddings(list(transcripts))
        
        return X_bert, groups
    
    elif modality == 'combined':
        print(f"\n{'='*60}")
        print("Using COMBINED (Prosody + BERT) features")
        print(f"{'='*60}")
        
        X_bert = get_bert_embeddings(list(transcripts))
        
        X_combined = np.hstack([X_prosody, X_bert])
        print(f"Combined feature shape: {X_combined.shape}")
        
        return X_combined, groups

def calculate_per_class_auroc(y_true, y_pred_proba, label_names):
    """Calculate AUROC for each class"""
    aurocs = []
    for i, label in enumerate(label_names):
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            aurocs.append(auroc)
        except:
            aurocs.append(np.nan)
    return aurocs

def calculate_per_class_kappa(y_true, y_pred, label_names):
    """Calculate Cohen's Kappa for each class"""
    kappas = []
    for i, label in enumerate(label_names):
        try:
            kappa = cohen_kappa_score(y_true[:, i], y_pred[:, i])
            kappas.append(kappa)
        except:
            kappas.append(np.nan)
    return kappas

def train_with_logo_cv(X, y, groups, label_names):
    """Train with Leave-One-Group-Out Cross-Validation"""
    
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    
    print(f"\n{'='*60}")
    print(f"Starting Leave-One-Group-Out CV ({n_splits} folds)")
    print(f"{'='*60}")
    
    # Store results
    results = {
        'hamming_loss': [],
        'subset_accuracy': [],
        'f1_micro': [],
        'f1_macro': [],
        'f1_weighted': [],
        'f1_samples': [],
        'auroc_macro': [],
        'kappa_macro': []
    }
    
    # Store per-class results
    per_class_auroc = {label: [] for label in label_names}
    per_class_kappa = {label: [] for label in label_names}
    
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups):
        test_group = groups[test_idx][0]
        print(f"\nFold {fold}/{n_splits} - Testing on group {test_group}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        base_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        clf = MultiOutputClassifier(base_clf)
        clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        
        # Get prediction probabilities (handle case where class might not exist)
        y_pred_proba = []
        for estimator in clf.estimators_:
            proba = estimator.predict_proba(X_test_scaled)
            # If only one class, proba will be (n_samples, 1), we need (n_samples, 2)
            if proba.shape[1] == 1:
                # Only negative class exists, so positive class probability is 0
                proba_class_1 = np.zeros(proba.shape[0])
            else:
                proba_class_1 = proba[:, 1]
            y_pred_proba.append(proba_class_1)
        
        y_pred_proba = np.array(y_pred_proba).T
        
        # Calculate metrics
        results['hamming_loss'].append(hamming_loss(y_test, y_pred))
        results['subset_accuracy'].append(accuracy_score(y_test, y_pred))
        results['f1_micro'].append(f1_score(y_test, y_pred, average='micro', zero_division=0))
        results['f1_macro'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        results['f1_weighted'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        results['f1_samples'].append(f1_score(y_test, y_pred, average='samples', zero_division=0))
        
        # Per-class AUROC
        aurocs = calculate_per_class_auroc(y_test, y_pred_proba, label_names)
        for i, label in enumerate(label_names):
            per_class_auroc[label].append(aurocs[i])
        results['auroc_macro'].append(np.nanmean(aurocs))
        
        # Per-class Kappa
        kappas = calculate_per_class_kappa(y_test, y_pred, label_names)
        for i, label in enumerate(label_names):
            per_class_kappa[label].append(kappas[i])
        results['kappa_macro'].append(np.nanmean(kappas))
        
        print(f"  F1 (samples): {results['f1_samples'][-1]:.3f}")
        print(f"  AUROC (macro): {results['auroc_macro'][-1]:.3f}")
        
        fold += 1
    
    # Calculate mean and std
    print(f"\n{'='*60}")
    print("Cross-Validation Results (Mean ± Std)")
    print(f"{'='*60}")
    
    for metric in results.keys():
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric:20s}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Per-class results
    print(f"\n{'='*60}")
    print("Per-Class AUROC (Mean ± Std)")
    print(f"{'='*60}")
    for label in label_names:
        scores = per_class_auroc[label]
        mean_val = np.nanmean(scores)
        std_val = np.nanstd(scores)
        marker = ' ⭐' if mean_val >= 0.6 else ''
        print(f"{label:20s}: {mean_val:.3f} ± {std_val:.3f}{marker}")
    
    print(f"\n{'='*60}")
    print("Per-Class Kappa (Mean ± Std)")
    print(f"{'='*60}")
    for label in label_names:
        scores = per_class_kappa[label]
        mean_val = np.nanmean(scores)
        std_val = np.nanstd(scores)
        print(f"{label:20s}: {mean_val:.3f} ± {std_val:.3f}")
    
    return results, per_class_auroc, per_class_kappa

def save_results(results_all, label_names, window_size, output_dir='results'):
    """Save comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create summary DataFrame
    summary_data = []
    for modality, (results, per_class_auroc, per_class_kappa) in results_all.items():
        row = {
            'WindowSize': window_size,
            'Modality': modality,
            'F1_samples_mean': np.mean(results['f1_samples']),
            'F1_samples_std': np.std(results['f1_samples']),
            'AUROC_macro_mean': np.mean(results['auroc_macro']),
            'AUROC_macro_std': np.std(results['auroc_macro']),
            'Kappa_macro_mean': np.mean(results['kappa_macro']),
            'Kappa_macro_std': np.std(results['kappa_macro']),
            'Hamming_loss_mean': np.mean(results['hamming_loss']),
            'Hamming_loss_std': np.std(results['hamming_loss']),
        }
        
        # Add per-class AUROC
        for label in label_names:
            row[f'AUROC_{label}_mean'] = np.nanmean(per_class_auroc[label])
            row[f'AUROC_{label}_std'] = np.nanstd(per_class_auroc[label])
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    output_file = os.path.join(output_dir, f'with_none_class_{window_size}_comparison_{timestamp}.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python multilabel_with_none_class.py <input_file> [window_size]")
        print("Example: python multilabel_with_none_class.py reports_agg_10s_windows_transcripts.csv 10s")
        sys.exit(1)
    
    input_file = sys.argv[1]
    window_size = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(input_file).split('_')[2]
    
    # Determine file type
    file_type = 'excel' if input_file.endswith('.xlsx') else 'csv'
    
    print(f"\n{'='*60}")
    print(f"Multi-label Affective State Prediction with None Class")
    print(f"Window Size: {window_size}")
    print(f"Input File: {input_file}")
    print(f"{'='*60}")
    
    # Load and prepare data
    X_prosody, y, groups, transcripts, label_names = prepare_multilabel_data(input_file, file_type)
    
    # Run experiments for each modality
    results_all = {}
    
    for modality in ['prosody', 'text', 'combined']:
        print(f"\n\n{'#'*60}")
        print(f"# EXPERIMENT: {modality.upper()}")
        print(f"{'#'*60}")
        
        # Prepare features
        X, groups_out = prepare_features(X_prosody, transcripts, groups, modality=modality)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Label matrix shape: {y.shape}")
        
        # Train and evaluate
        results, per_class_auroc, per_class_kappa = train_with_logo_cv(X, y, groups_out, label_names)
        
        results_all[modality] = (results, per_class_auroc, per_class_kappa)
    
    # Save results
    output_file = save_results(results_all, label_names, window_size)
    
    # Print final comparison
    print(f"\n\n{'='*60}")
    print(f"FINAL COMPARISON - Window Size: {window_size} (with None class)")
    print(f"{'='*60}")
    
    for modality, (results, _, _) in results_all.items():
        f1_mean = np.mean(results['f1_samples'])
        f1_std = np.std(results['f1_samples'])
        auroc_mean = np.mean(results['auroc_macro'])
        auroc_std = np.std(results['auroc_macro'])
        
        print(f"\n{modality.upper():12s}: F1={f1_mean:.3f}±{f1_std:.3f}, AUROC={auroc_mean:.3f}±{auroc_std:.3f}")

if __name__ == "__main__":
    main()
