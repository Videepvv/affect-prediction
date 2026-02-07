"""
Valence-based classification for cognitive-affective state detection.

Groups states into Positive (Optimistic, Curious) vs Negative (Confused,
Frustrated, Conflicted, Disengaged, Surprised) and runs binary classification.

Also tests PCA-reduced combined features to address the dimensionality
imbalance between prosodic (88d) and transcript embedding (768d) features.

Features: Prosodic, Transcript, Combined (raw concat), Combined-PCA
Models:   Logistic Regression, Random Forest, SVM
Eval:     AUROC with Leave-One-Group-Out CV, per-group breakdown

Usage:
    python valence_classification.py data.csv
    python valence_classification.py data.csv -f all --seeds 10
    python valence_classification.py data.csv --pca-dims 50 100 200
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
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Valence groupings
# ─────────────────────────────────────────────────────────────

POSITIVE_STATES = ['Optimistic', 'Curious']
NEGATIVE_STATES = ['Confused', 'Frustrated', 'Conflicted', 'Disengaged', 'Surprised']

PROSODIC_KEYWORDS = [
    'F0', 'loudness', 'spectral', 'mfcc', 'jitter', 'shimmer',
    'HNR', 'logRel', 'frequency', 'bandwidth', 'amplitude',
    'alpha', 'hammarberg', 'slope', 'Voiced', 'Unvoiced',
    'equivalentSoundLevel', 'Peaks'
]


# ─────────────────────────────────────────────────────────────
# Feature extraction (same as per-state script)
# ─────────────────────────────────────────────────────────────

def get_prosodic_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns
            if any(kw in col for kw in PROSODIC_KEYWORDS)]


def get_prosodic_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = get_prosodic_columns(df)
    features = df[cols].values.astype(np.float64)
    col_means = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        mask = np.isnan(features[:, i])
        features[mask, i] = col_means[i]
    print(f"  Prosodic features: {features.shape[1]} columns")
    return features, cols


def get_sentence_embeddings(texts: List[str],
                            model_name: str = 'all-mpnet-base-v2',
                            batch_size: int = 64) -> np.ndarray:
    print(f"  Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
    print(f"  Encoding {len(clean_texts)} texts...")
    embeddings = model.encode(clean_texts, batch_size=batch_size,
                              show_progress_bar=True, convert_to_numpy=True)
    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────────────────────
# Data preparation for valence
# ─────────────────────────────────────────────────────────────

def build_valence_dataset(df: pd.DataFrame, mode: str = 'pos_vs_neg'
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build a valence-based binary dataset.

    Modes:
        'pos_vs_neg':      Positive states vs Negative states (labeled only)
        'pos_vs_unlabeled': Positive states vs unlabeled (balanced 1:1)
        'neg_vs_unlabeled': Negative states vs unlabeled (balanced 1:1)

    Returns:
        indices, labels (1=positive class, 0=negative class), groups, info dict
    """
    pos_mask = df['label1'].isin(POSITIVE_STATES)
    neg_mask = df['label1'].isin(NEGATIVE_STATES)
    unlabeled_mask = df['label1'].isna()

    if mode == 'pos_vs_neg':
        pos_idx = df.index[pos_mask].values
        neg_idx = df.index[neg_mask].values
        info = {
            'pos_label': 'Positive',
            'neg_label': 'Negative',
            'n_pos': len(pos_idx),
            'n_neg': len(neg_idx),
        }
    elif mode == 'pos_vs_unlabeled':
        pos_idx = df.index[pos_mask].values
        neg_pool = df.index[unlabeled_mask].values
        # Balance 1:1
        rng = np.random.default_rng(42)
        neg_idx = rng.choice(neg_pool, size=min(len(pos_idx), len(neg_pool)),
                             replace=False)
        info = {
            'pos_label': 'Positive',
            'neg_label': 'Unlabeled',
            'n_pos': len(pos_idx),
            'n_neg': len(neg_idx),
        }
    elif mode == 'neg_vs_unlabeled':
        pos_idx = df.index[neg_mask].values  # "positive class" = negative valence
        neg_pool = df.index[unlabeled_mask].values
        rng = np.random.default_rng(42)
        neg_idx = rng.choice(neg_pool, size=min(len(pos_idx), len(neg_pool)),
                             replace=False)
        info = {
            'pos_label': 'Negative-valence',
            'neg_label': 'Unlabeled',
            'n_pos': len(pos_idx),
            'n_neg': len(neg_idx),
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    indices = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    groups = df.loc[indices, 'groupID'].values

    return indices, labels, groups, info


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

def get_models() -> Dict[str, object]:
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
# Evaluation (with per-group + optional PCA inside CV)
# ─────────────────────────────────────────────────────────────

def evaluate_logo_binary(X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray, model,
                         pca_n: Optional[int] = None,
                         prosodic_dim: Optional[int] = None
                         ) -> Dict[str, object]:
    """
    LOGO CV with optional PCA on the transcript portion of combined features.

    If pca_n is set and prosodic_dim is set, applies PCA to the transcript
    portion (columns prosodic_dim:) within each fold, then re-concatenates.
    PCA is fit on train only to prevent leakage.
    """
    logo = LeaveOneGroupOut()
    all_y_true, all_y_prob, all_y_pred = [], [], []
    per_group_auroc = {}

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_te = groups[test_idx]

        if pca_n is not None and prosodic_dim is not None:
            # Split into prosodic and transcript portions
            X_tr_pros = X_tr[:, :prosodic_dim]
            X_tr_emb = X_tr[:, prosodic_dim:]
            X_te_pros = X_te[:, :prosodic_dim]
            X_te_emb = X_te[:, prosodic_dim:]

            # PCA on transcript embeddings (fit on train only)
            n_components = min(pca_n, X_tr_emb.shape[1], X_tr_emb.shape[0])
            pca = PCA(n_components=n_components, random_state=42)
            X_tr_emb_pca = pca.fit_transform(X_tr_emb)
            X_te_emb_pca = pca.transform(X_te_emb)

            X_tr = np.hstack([X_tr_pros, X_tr_emb_pca])
            X_te = np.hstack([X_te_pros, X_te_emb_pca])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

        all_y_true.append(y_te)
        all_y_prob.append(y_prob)
        all_y_pred.append(y_pred)

        group_id = g_te[0]
        if len(np.unique(y_te)) > 1:
            per_group_auroc[group_id] = roc_auc_score(y_te, y_prob)
        else:
            per_group_auroc[group_id] = np.nan

    y_true = np.concatenate(all_y_true)
    y_prob = np.concatenate(all_y_prob)
    y_pred = np.concatenate(all_y_pred)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    return {
        'auroc': auroc,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'per_group_auroc': per_group_auroc,
    }


# ─────────────────────────────────────────────────────────────
# Feature precomputation
# ─────────────────────────────────────────────────────────────

def precompute_features(df: pd.DataFrame, feature_type: str,
                        embedding_model: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Returns (features_dict, prosodic_dim) where prosodic_dim is needed
    for PCA-based combined features.
    """
    features = {}
    prosodic_dim = 0

    if feature_type in ('prosodic', 'all'):
        feats, _ = get_prosodic_features(df)
        features['prosodic'] = feats
        prosodic_dim = feats.shape[1]

    if feature_type in ('transcript', 'all'):
        emb = get_sentence_embeddings(df['transcript'].tolist(),
                                      model_name=embedding_model)
        features['transcript'] = emb

    if feature_type == 'all':
        features['combined'] = np.hstack([features['prosodic'],
                                          features['transcript']])
        print(f"  Combined features: {features['combined'].shape[1]} columns "
              f"({prosodic_dim} prosodic + {features['transcript'].shape[1]} embedding)")

    return features, prosodic_dim


# ─────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────

def run_valence_experiments(df: pd.DataFrame,
                            feature_type: str = 'all',
                            embedding_model: str = 'all-mpnet-base-v2',
                            pca_dims: Optional[List[int]] = None,
                            modes: Optional[List[str]] = None,
                            ) -> pd.DataFrame:
    """
    Run valence classification experiments.

    Tests all combinations of:
      - Valence modes (pos_vs_neg, pos_vs_unlabeled, neg_vs_unlabeled)
      - Feature sets (prosodic, transcript, combined, combined-pca-N)
      - Models (LR, RF, SVM)
    """
    if pca_dims is None:
        pca_dims = [50, 100]
    if modes is None:
        modes = ['pos_vs_neg', 'pos_vs_unlabeled', 'neg_vs_unlabeled']

    print("=" * 60)
    print("Valence-Based Affect Classification")
    print("=" * 60)

    # Precompute features
    print("\nPrecomputing features...")
    all_features, prosodic_dim = precompute_features(df, feature_type,
                                                     embedding_model)

    feat_names = (
        [feature_type] if feature_type in ('prosodic', 'transcript')
        else ['prosodic', 'transcript', 'combined']
    )

    models_dict = get_models()
    results = []

    for mode in modes:
        indices, labels, groups, info = build_valence_dataset(df, mode)

        print(f"\n{'═' * 60}")
        print(f"Mode: {mode}")
        print(f"  {info['pos_label']}: {info['n_pos']}  |  "
              f"{info['neg_label']}: {info['n_neg']}")
        print(f"  Groups: {sorted(np.unique(groups))}")
        print(f"{'═' * 60}")

        for feat_name in feat_names:
            X = all_features[feat_name][indices]

            # Determine PCA configs for this feature set
            pca_configs = [(feat_name, None)]  # always run without PCA
            if feat_name == 'combined' and feature_type == 'all':
                for n in pca_dims:
                    pca_configs.append((f'combined-pca-{n}', n))

            for display_name, pca_n in pca_configs:
                for model_name in models_dict:
                    model = get_models()[model_name]

                    metrics = evaluate_logo_binary(
                        X, labels, groups, model,
                        pca_n=pca_n,
                        prosodic_dim=prosodic_dim if pca_n else None,
                    )

                    result_row = {
                        'mode': mode,
                        'pos_class': info['pos_label'],
                        'neg_class': info['neg_label'],
                        'n_pos': info['n_pos'],
                        'n_neg': info['n_neg'],
                        'feature_type': display_name,
                        'model': model_name,
                        'auroc': metrics['auroc'],
                        'f1': metrics['f1'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                    }
                    # Per-group
                    for gid, val in sorted(metrics['per_group_auroc'].items()):
                        result_row[f'auroc_group_{gid}'] = val

                    results.append(result_row)

                    # Console
                    gstr = "  ".join(
                        f"G{gid}={v:.2f}" if not np.isnan(v) else f"G{gid}=n/a"
                        for gid, v in sorted(metrics['per_group_auroc'].items())
                    )
                    print(f"  {display_name:20s} | {model_name:20s} | "
                          f"AUROC={metrics['auroc']:.3f}  "
                          f"F1={metrics['f1']:.3f}")
                    print(f"  {'':20s} | {'per-group':20s} | {gstr}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valence-based cognitive-affective state classification"
    )
    parser.add_argument("input_file", help="Path to preprocessed CSV")
    parser.add_argument(
        "-f", "--features",
        choices=['prosodic', 'transcript', 'all'],
        default='all',
        help="Feature type (default: all)"
    )
    parser.add_argument(
        "--modes", nargs='+',
        default=['pos_vs_neg', 'pos_vs_unlabeled', 'neg_vs_unlabeled'],
        help="Valence modes to test"
    )
    parser.add_argument(
        "--pca-dims", type=int, nargs='+', default=[50, 100],
        help="PCA dimensions for reduced combined features (default: 50 100)"
    )
    parser.add_argument(
        "--embedding-model", default='all-mpnet-base-v2',
        help="SentenceTransformer model (default: all-mpnet-base-v2)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path"
    )

    args = parser.parse_args()

    print(f"Loading: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples")

    # Show valence distribution
    labeled = df[df['label1'].notna()]
    n_pos = labeled['label1'].isin(POSITIVE_STATES).sum()
    n_neg = labeled['label1'].isin(NEGATIVE_STATES).sum()
    print(f"Positive states ({', '.join(POSITIVE_STATES)}): {n_pos}")
    print(f"Negative states ({', '.join(NEGATIVE_STATES)}): {n_neg}")
    print(f"Unlabeled: {df['label1'].isna().sum()}")

    results_df = run_valence_experiments(
        df,
        feature_type=args.features,
        embedding_model=args.embedding_model,
        pca_dims=args.pca_dims,
        modes=args.modes,
    )

    if args.output is None:
        p = Path(args.input_file)
        output_path = p.parent / f"valence_results_{args.features}.csv"
    else:
        output_path = args.output

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode in results_df['mode'].unique():
        sub = results_df[results_df['mode'] == mode]
        print(f"\n  Mode: {mode}")
        pivot = sub.pivot_table(
            index='feature_type', columns='model',
            values='auroc', aggfunc='first'
        )
        print(pivot.to_string(float_format='{:.3f}'.format))

    # Highlight best per mode
    print("\n" + "=" * 60)
    print("BEST CONFIG PER MODE")
    print("=" * 60)
    for mode in results_df['mode'].unique():
        sub = results_df[results_df['mode'] == mode]
        best = sub.loc[sub['auroc'].idxmax()]
        print(f"  {mode:25s}: AUROC={best['auroc']:.3f}  "
              f"({best['feature_type']}, {best['model']})")


if __name__ == "__main__":
    main()