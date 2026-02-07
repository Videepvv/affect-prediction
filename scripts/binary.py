"""
Binary one-vs-rest classification for cognitive-affective state detection.

For each target state, trains a binary classifier:
  Positive = samples labeled with that state (label1)
  Negative = randomly sampled unlabeled (no-state) windows, balanced 1:1

Features: Prosodic (eGeMAPS), BERT embeddings, Combined
Models:   Logistic Regression, Random Forest, SVM
Eval:     AUROC with Leave-One-Group-Out CV
          Multiple resampling seeds to reduce variance from negative sampling

Usage:
    python binary_affect_classification.py data.csv
    python binary_affect_classification.py data.csv -f prosodic --seeds 10
    python binary_affect_classification.py data.csv --states Confused Curious Optimistic
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertModel
import torch

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

PROSODIC_KEYWORDS = [
    'F0', 'loudness', 'spectral', 'mfcc', 'jitter', 'shimmer',
    'HNR', 'logRel', 'frequency', 'bandwidth', 'amplitude',
    'alpha', 'hammarberg', 'slope', 'Voiced', 'Unvoiced',
    'equivalentSoundLevel', 'Peaks'
]


def get_prosodic_columns(df: pd.DataFrame) -> List[str]:
    """Identify prosodic feature columns by keyword matching."""
    return [col for col in df.columns
            if any(kw in col for kw in PROSODIC_KEYWORDS)]


def get_prosodic_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract prosodic features, imputing NaNs with column means."""
    cols = get_prosodic_columns(df)
    features = df[cols].values.astype(np.float64)

    # Impute NaN with column mean
    col_means = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        mask = np.isnan(features[:, i])
        features[mask, i] = col_means[i]

    print(f"  Prosodic features: {features.shape[1]} columns")
    return features, cols


def get_bert_embeddings(texts: List[str],
                        model_name: str = 'bert-base-uncased',
                        batch_size: int = 32) -> np.ndarray:
    """
    Extract [CLS] token embeddings from BERT.
    Returns array of shape (n_samples, 768).
    """
    print(f"  Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model = model.to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) if pd.notna(t) else "" for t in texts[i:i + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)
            embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())

        if (i + batch_size) % 200 == 0 or i + batch_size >= len(texts):
            print(f"    BERT: {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(embeddings)


# ─────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────

def build_binary_dataset(df: pd.DataFrame,
                         target_state: str,
                         rng: np.random.Generator
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a balanced binary dataset for one target state.

    Positives: rows where label1 == target_state
    Negatives: randomly sampled unlabeled rows (label1 is NaN), 1:1 ratio

    Returns:
        indices: row indices into df for the selected samples
        labels:  binary labels (1 = target state, 0 = no state)
        groups:  groupID for each selected sample
    """
    pos_mask = df['label1'] == target_state
    neg_pool_mask = df['label1'].isna()

    pos_idx = df.index[pos_mask].values
    neg_pool_idx = df.index[neg_pool_mask].values

    n_pos = len(pos_idx)

    # Sample negatives 1:1, stratified by group to preserve group structure
    # If not enough unlabeled in a group, take what's available
    neg_idx = rng.choice(neg_pool_idx, size=min(n_pos, len(neg_pool_idx)),
                         replace=False)

    indices = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    groups = df.loc[indices, 'groupID'].values

    return indices, labels, groups


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

def get_models() -> Dict[str, object]:
    """Binary classifiers to evaluate."""
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42,
            class_weight='balanced'
        ),
    }


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_logo_binary(X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray,
                         model) -> Dict[str, float]:
    """
    Leave-One-Group-Out CV for a binary classifier.
    Returns AUROC, F1, precision, recall (macro across folds).
    """
    logo = LeaveOneGroupOut()
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

        all_y_true.append(y_te)
        all_y_prob.append(y_prob)
        all_y_pred.append(y_pred)

    y_true = np.concatenate(all_y_true)
    y_prob = np.concatenate(all_y_prob)
    y_pred = np.concatenate(all_y_pred)

    # AUROC only if both classes present
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
    else:
        auroc = np.nan

    return {
        'auroc': auroc,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────

def precompute_features(df: pd.DataFrame, feature_type: str,
                        bert_model: str) -> Dict[str, np.ndarray]:
    """
    Precompute all feature matrices once over the full dataframe.
    Returns dict mapping feature_name -> (n_samples, n_features) array.
    """
    features = {}

    if feature_type in ('prosodic', 'all'):
        feats, _ = get_prosodic_features(df)
        features['prosodic'] = feats

    if feature_type in ('transcript', 'all'):
        emb = get_bert_embeddings(df['transcript'].tolist(),
                                  model_name=bert_model)
        features['transcript'] = emb

    if feature_type == 'all':
        features['combined'] = np.hstack([features['prosodic'],
                                          features['transcript']])
        print(f"  Combined features: {features['combined'].shape[1]} columns")

    return features


def run_experiments(df: pd.DataFrame,
                    feature_type: str = 'all',
                    bert_model: str = 'bert-base-uncased',
                    target_states: Optional[List[str]] = None,
                    n_seeds: int = 5) -> pd.DataFrame:
    """
    Run binary one-vs-rest experiments for each target state.

    For each state × feature_set × model × seed:
      1. Sample balanced binary dataset (pos = state, neg = unlabeled)
      2. Run LOGO CV
      3. Record metrics

    Final results are averaged across seeds.
    """
    print("=" * 60)
    print("Binary One-vs-Rest Affect Classification")
    print("=" * 60)

    # Determine target states
    all_states = df['label1'].value_counts()
    if target_states is None:
        target_states = all_states.index.tolist()

    print(f"\nTarget states: {target_states}")
    print(f"Resampling seeds: {n_seeds}")
    for state in target_states:
        print(f"  {state}: {all_states[state]} positive samples")
    print(f"Unlabeled pool: {df['label1'].isna().sum()} samples")

    # Precompute features on the full dataframe
    print("\nPrecomputing features...")
    all_features = precompute_features(df, feature_type, bert_model)

    feat_names = (
        [feature_type] if feature_type in ('prosodic', 'transcript')
        else ['prosodic', 'transcript', 'combined']
    )

    models_dict = get_models()
    results = []

    for state in target_states:
        print(f"\n{'─' * 50}")
        print(f"State: {state} (n={all_states[state]})")
        print(f"{'─' * 50}")

        for feat_name in feat_names:
            X_full = all_features[feat_name]

            for model_name in models_dict:
                seed_metrics = []

                for seed in range(n_seeds):
                    rng = np.random.default_rng(seed)
                    indices, y, groups = build_binary_dataset(df, state, rng)

                    X = X_full[indices]

                    # Fresh model instance each time
                    model = get_models()[model_name]

                    metrics = evaluate_logo_binary(X, y, groups, model)
                    seed_metrics.append(metrics)

                # Average across seeds
                avg = {k: np.nanmean([m[k] for m in seed_metrics])
                       for k in seed_metrics[0]}
                std = {k: np.nanstd([m[k] for m in seed_metrics])
                       for k in seed_metrics[0]}

                results.append({
                    'state': state,
                    'n_positive': all_states[state],
                    'feature_type': feat_name,
                    'model': model_name,
                    'auroc_mean': avg['auroc'],
                    'auroc_std': std['auroc'],
                    'f1_mean': avg['f1'],
                    'f1_std': std['f1'],
                    'precision_mean': avg['precision'],
                    'precision_std': std['precision'],
                    'recall_mean': avg['recall'],
                    'recall_std': std['recall'],
                })

                print(f"  {feat_name:12s} | {model_name:20s} | "
                      f"AUROC={avg['auroc']:.3f}±{std['auroc']:.3f}  "
                      f"F1={avg['f1']:.3f}±{std['f1']:.3f}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Binary one-vs-rest cognitive-affective state classification"
    )
    parser.add_argument("input_file", help="Path to preprocessed CSV")
    parser.add_argument(
        "-f", "--features",
        choices=['prosodic', 'transcript', 'all'],
        default='all',
        help="Feature type (default: all → prosodic + transcript + combined)"
    )
    parser.add_argument(
        "--states", nargs='+', default=None,
        help="Target states to classify (default: all states in data)"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of negative resampling seeds (default: 5)"
    )
    parser.add_argument(
        "--bert-model", default='bert-base-uncased',
        help="BERT model name (default: bert-base-uncased)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: auto-generated)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples "
          f"({df['label1'].notna().sum()} labeled, "
          f"{df['label1'].isna().sum()} unlabeled)")

    # Run
    results_df = run_experiments(
        df,
        feature_type=args.features,
        bert_model=args.bert_model,
        target_states=args.states,
        n_seeds=args.seeds,
    )

    # Save
    if args.output is None:
        p = Path(args.input_file)
        
        # Extract window size from filename (e.g., "10s", "5s", "20s")
        window_size = "unknown"
        for part in p.stem.split('_'):
            if part.endswith('s') and part[:-1].isdigit():
                window_size = part
                break
        
        # Get script name (without .py extension)
        script_name = Path(__file__).stem  # e.g., "binary"
        
        # Create results folder
        results_dir = p.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Descriptive filename: scriptname_windowsize_featuretype_results.csv
        output_path = results_dir / f"{script_name}_{window_size}_{args.features}_results.csv"
    else:
        output_path = Path(args.output)

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY (AUROC mean ± std across resampling seeds)")
    print("=" * 60)
    pivot = results_df.pivot_table(
        index=['state', 'feature_type'],
        columns='model',
        values='auroc_mean',
        aggfunc='first'
    )
    print(pivot.to_string(float_format='{:.3f}'.format))


if __name__ == "__main__":
    main()