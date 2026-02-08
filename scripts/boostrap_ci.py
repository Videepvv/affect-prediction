"""
Bootstrap confidence intervals and permutation tests for binary affect classification.

Runs the same pipeline as binary_affect_classification.py but adds:
  1. Bootstrap 95% CIs on pooled LOGO AUROC (resampling test predictions)
  2. Permutation test: shuffle labels 1000x, compute null AUROC distribution,
     report p-value for observed AUROC > chance

Uses the SAME code structure: binary one-vs-rest, balanced 1:1, LOGO CV,
5 resampling seeds. Only runs best config per state (from prior results).

Usage:
    python bootstrap_ci.py data.csv
    python bootstrap_ci.py data.csv --n-bootstrap 2000 --n-permutations 1000
    python bootstrap_ci.py data.csv --states Curious Optimistic
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Feature extraction (same as main script)
# ─────────────────────────────────────────────────────────────

PROSODIC_KEYWORDS = [
    'F0', 'loudness', 'spectral', 'mfcc', 'jitter', 'shimmer',
    'HNR', 'logRel', 'frequency', 'bandwidth', 'amplitude',
    'alpha', 'hammarberg', 'slope', 'Voiced', 'Unvoiced',
    'equivalentSoundLevel', 'Peaks'
]


def get_prosodic_columns(df):
    return [col for col in df.columns
            if any(kw in col for kw in PROSODIC_KEYWORDS)]


def get_prosodic_features(df):
    cols = get_prosodic_columns(df)
    features = df[cols].values.astype(np.float64)
    col_means = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        mask = np.isnan(features[:, i])
        features[mask, i] = col_means[i]
    return features


def get_sentence_embeddings(texts, model_name='all-mpnet-base-v2'):
    print(f"  Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
    print(f"  Encoding {len(clean_texts)} texts...")
    return model.encode(clean_texts, batch_size=128, show_progress_bar=True,
                        convert_to_numpy=True, device='cuda')


def build_binary_dataset(df, target_state, rng):
    pos_mask = df['label1'] == target_state
    neg_pool_mask = df['label1'].isna()
    pos_idx = df.index[pos_mask].values
    neg_pool_idx = df.index[neg_pool_mask].values
    n_pos = len(pos_idx)
    neg_idx = rng.choice(neg_pool_idx, size=min(n_pos, len(neg_pool_idx)),
                         replace=False)
    indices = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    groups = df.loc[indices, 'groupID'].values
    return indices, labels, groups


# ─────────────────────────────────────────────────────────────
# LOGO CV that returns raw predictions (not just metrics)
# ─────────────────────────────────────────────────────────────

def logo_cv_predictions(X, y, groups, model_class, model_kwargs):
    """
    Run LOGO CV and return pooled (y_true, y_prob, group_ids) arrays.
    These raw predictions are what we bootstrap over.
    """
    logo = LeaveOneGroupOut()
    all_y_true, all_y_prob, all_groups = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_te = groups[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]

        all_y_true.append(y_te)
        all_y_prob.append(y_prob)
        all_groups.append(g_te)

    return (np.concatenate(all_y_true),
            np.concatenate(all_y_prob),
            np.concatenate(all_groups))


# ─────────────────────────────────────────────────────────────
# Bootstrap CI on AUROC
# ─────────────────────────────────────────────────────────────

def bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=2000, ci=95, seed=42):
    """
    Compute bootstrap confidence interval on AUROC.
    Resamples (y_true, y_prob) pairs with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_prob[idx]
        # Need both classes in bootstrap sample
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))

    aurocs = np.array(aurocs)
    lower = np.percentile(aurocs, (100 - ci) / 2)
    upper = np.percentile(aurocs, 100 - (100 - ci) / 2)
    return {
        'mean': np.mean(aurocs),
        'median': np.median(aurocs),
        'ci_lower': lower,
        'ci_upper': upper,
        'std': np.std(aurocs),
        'n_valid': len(aurocs),
    }


# ─────────────────────────────────────────────────────────────
# Permutation test
# ─────────────────────────────────────────────────────────────

def permutation_test_auroc(X, y, groups, model_class, model_kwargs,
                           observed_auroc, n_permutations=1000, seed=42):
    """
    Permutation test: shuffle labels, rerun LOGO CV, build null distribution.
    p-value = proportion of null AUROCs >= observed AUROC.
    """
    rng = np.random.default_rng(seed)
    null_aurocs = []

    for i in range(n_permutations):
        # Shuffle labels while keeping group structure intact
        y_perm = rng.permutation(y)

        try:
            y_true_perm, y_prob_perm, _ = logo_cv_predictions(
                X, y_perm, groups, model_class, model_kwargs
            )
            if len(np.unique(y_true_perm)) > 1:
                null_aurocs.append(roc_auc_score(y_true_perm, y_prob_perm))
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"      Permutation {i+1}/{n_permutations}")

    null_aurocs = np.array(null_aurocs)
    p_value = np.mean(null_aurocs >= observed_auroc)

    return {
        'p_value': p_value,
        'null_mean': np.mean(null_aurocs),
        'null_std': np.std(null_aurocs),
        'n_valid': len(null_aurocs),
    }


# ─────────────────────────────────────────────────────────────
# Best configs from prior results
# ─────────────────────────────────────────────────────────────

# These are the best configs from the 20s results
BEST_CONFIGS = {
    'Optimistic':  {'feature': 'transcript', 'model': 'LogisticRegression'},
    'Confused':    {'feature': 'prosodic',   'model': 'LogisticRegression'},
    'Curious':     {'feature': 'prosodic',   'model': 'LogisticRegression'},
    'Disengaged':  {'feature': 'prosodic',   'model': 'LogisticRegression'},
    'Surprised':   {'feature': 'prosodic',   'model': 'LogisticRegression'},
    'Frustrated':  {'feature': 'transcript', 'model': 'LogisticRegression'},
    'Conflicted':  {'feature': 'prosodic',   'model': 'SVM'},
}

MODEL_REGISTRY = {
    'LogisticRegression': (LogisticRegression,
                           {'max_iter': 1000, 'random_state': 42,
                            'class_weight': 'balanced'}),
    'SVM': (SVC,
            {'kernel': 'rbf', 'probability': True, 'random_state': 42,
             'class_weight': 'balanced'}),
}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs and permutation tests for affect classification"
    )
    parser.add_argument("input_file", help="Path to preprocessed CSV")
    parser.add_argument("--states", nargs='+', default=None,
                        help="States to test (default: all)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Negative resampling seeds (default: 5)")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Bootstrap iterations (default: 2000)")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Permutation test iterations (default: 1000)")
    parser.add_argument("--embedding-model", default='all-mpnet-base-v2')
    parser.add_argument("--skip-permutation", action='store_true',
                        help="Skip permutation test (slow)")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    print(f"Loading: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples")

    # Precompute features
    print("\nPrecomputing features...")
    features = {}
    needs_prosodic = any(BEST_CONFIGS[s]['feature'] in ('prosodic', 'combined')
                         for s in (args.states or BEST_CONFIGS.keys()))
    needs_transcript = any(BEST_CONFIGS[s]['feature'] in ('transcript', 'combined')
                           for s in (args.states or BEST_CONFIGS.keys()))

    if needs_prosodic:
        features['prosodic'] = get_prosodic_features(df)
        print(f"  Prosodic: {features['prosodic'].shape}")
    if needs_transcript:
        features['transcript'] = get_sentence_embeddings(
            df['transcript'].tolist(), model_name=args.embedding_model)
        print(f"  Transcript: {features['transcript'].shape}")

    target_states = args.states or list(BEST_CONFIGS.keys())
    results = []

    for state in target_states:
        config = BEST_CONFIGS[state]
        feat_name = config['feature']
        model_name = config['model']
        model_class, model_kwargs = MODEL_REGISTRY[model_name]
        X_full = features[feat_name]

        print(f"\n{'═' * 60}")
        print(f"State: {state} ({feat_name}, {model_name})")
        print(f"{'═' * 60}")

        # Collect predictions across seeds
        all_seed_y_true = []
        all_seed_y_prob = []
        seed_aurocs = []

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            indices, y, groups = build_binary_dataset(df, state, rng)
            X = X_full[indices]

            y_true, y_prob, g = logo_cv_predictions(
                X, y, groups, model_class, model_kwargs
            )

            auroc = roc_auc_score(y_true, y_prob)
            seed_aurocs.append(auroc)
            all_seed_y_true.append(y_true)
            all_seed_y_prob.append(y_prob)
            print(f"  Seed {seed}: AUROC = {auroc:.3f}")

        # Pool predictions across all seeds for bootstrap
        pooled_y_true = np.concatenate(all_seed_y_true)
        pooled_y_prob = np.concatenate(all_seed_y_prob)
        observed_auroc = np.mean(seed_aurocs)

        print(f"\n  Observed AUROC: {observed_auroc:.3f} "
              f"(seed mean ± std: {np.mean(seed_aurocs):.3f} ± {np.std(seed_aurocs):.3f})")

        # Bootstrap CI on pooled predictions
        print(f"  Running {args.n_bootstrap} bootstrap iterations...")
        boot = bootstrap_auroc_ci(pooled_y_true, pooled_y_prob,
                                  n_bootstrap=args.n_bootstrap)
        print(f"  Bootstrap 95% CI: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
        print(f"  Bootstrap mean: {boot['mean']:.3f}, std: {boot['std']:.3f}")

        result_row = {
            'state': state,
            'feature_type': feat_name,
            'model': model_name,
            'auroc_observed': observed_auroc,
            'auroc_seed_std': np.std(seed_aurocs),
            'bootstrap_mean': boot['mean'],
            'bootstrap_std': boot['std'],
            'ci_lower_95': boot['ci_lower'],
            'ci_upper_95': boot['ci_upper'],
            'ci_excludes_0.5': boot['ci_lower'] > 0.5,
        }

        # Permutation test
        if not args.skip_permutation:
            print(f"  Running {args.n_permutations} permutations...")
            # Use seed 0 dataset for permutation test
            rng = np.random.default_rng(0)
            indices, y, groups = build_binary_dataset(df, state, rng)
            X = X_full[indices]

            perm = permutation_test_auroc(
                X, y, groups, model_class, model_kwargs,
                observed_auroc=seed_aurocs[0],  # compare against seed 0
                n_permutations=args.n_permutations,
            )
            print(f"  Permutation p-value: {perm['p_value']:.4f}")
            print(f"  Null distribution: {perm['null_mean']:.3f} ± {perm['null_std']:.3f}")

            result_row.update({
                'perm_p_value': perm['p_value'],
                'null_mean': perm['null_mean'],
                'null_std': perm['null_std'],
            })
        else:
            result_row.update({
                'perm_p_value': np.nan,
                'null_mean': np.nan,
                'null_std': np.nan,
            })

        results.append(result_row)

        # Summary
        sig_str = ""
        if not args.skip_permutation:
            sig_str = f"  p={perm['p_value']:.4f}"
        above = "YES" if boot['ci_lower'] > 0.5 else "NO"
        print(f"\n  ╔═ {state}: AUROC={observed_auroc:.3f} "
              f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}] "
              f"above chance: {above}{sig_str}")

    # Save
    results_df = pd.DataFrame(results)
    if args.output is None:
        p = Path(args.input_file)
        output_path = p.parent / "bootstrap_ci_results.csv"
    else:
        output_path = Path(args.output)

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'State':12s} | {'AUROC':6s} | {'95% CI':17s} | {'>0.5?':5s} | {'p-value':7s}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        p_str = f"{row['perm_p_value']:.4f}" if not np.isnan(row['perm_p_value']) else "---"
        above = "YES" if row['ci_excludes_0.5'] else "NO"
        print(f"{row['state']:12s} | {row['auroc_observed']:.3f}  | "
              f"[{row['ci_lower_95']:.3f}, {row['ci_upper_95']:.3f}] | "
              f"{above:5s} | {p_str}")


if __name__ == "__main__":
    main()