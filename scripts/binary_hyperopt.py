"""
Binary one-vs-rest classification with hyperparameter optimization.

Optimized for high-core-count machines with GPU.

For each target state:
  - Outer loop: Leave-One-Group-Out CV (evaluation)
  - Inner loop: Optuna hyperopt with stratified K-fold on training data
  - Optimizes: AUROC (threshold-independent)

Features: Prosodic (eGeMAPS), Sentence embeddings (SentenceTransformer), Combined
Models:   LR, Ridge, LinearSVC, RF, ExtraTrees, GBM, XGBoost (GPU), SVM
Eval:     AUROC + F1/P/R with LOGO CV, per-group breakdown

Usage:
    python binary_hyperopt.py data.csv
    python binary_hyperopt.py data.csv -f prosodic --trials 80
    python binary_hyperopt.py data.csv --states Curious Optimistic Disengaged Surprised
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import optuna
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import partial

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Check for XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
    print("XGBoost available (will use GPU if present)")
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, skipping. Install with: pip install xgboost")

# Detect GPU for XGBoost
import subprocess
HAS_GPU = False
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        HAS_GPU = True
        print("GPU detected, XGBoost will use gpu_hist")
except FileNotFoundError:
    pass

# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

PROSODIC_KEYWORDS = [
    'F0', 'loudness', 'spectral', 'mfcc', 'jitter', 'shimmer',
    'HNR', 'logRel', 'frequency', 'bandwidth', 'amplitude',
    'alpha', 'hammarberg', 'slope', 'Voiced', 'Unvoiced',
    'equivalentSoundLevel', 'Peaks'
]

N_JOBS = -1  # use all cores


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
                            batch_size: int = 128) -> np.ndarray:
    print(f"  Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
    print(f"  Encoding {len(clean_texts)} texts...")
    embeddings = model.encode(clean_texts, batch_size=batch_size,
                              show_progress_bar=True, convert_to_numpy=True,
                              device='cuda')
    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────

def build_binary_dataset(df: pd.DataFrame, target_state: str,
                         rng: np.random.Generator
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# Model construction from Optuna trial
# ─────────────────────────────────────────────────────────────

MODEL_NAMES = [
    'LogisticRegression', 'Ridge', 'LinearSVC',
    'RandomForest', 'ExtraTrees', 'GBM',
    'SVM',
]
if HAS_XGB:
    MODEL_NAMES.append('XGBoost')


def suggest_model(trial, n_features: int):
    """Suggest a model + hyperparameters from Optuna trial."""
    name = trial.suggest_categorical('model', MODEL_NAMES)

    if name == 'LogisticRegression':
        penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet'])
        return LogisticRegression(
            C=trial.suggest_float('lr_C', 1e-4, 100, log=True),
            penalty=penalty,
            l1_ratio=trial.suggest_float('lr_l1_ratio', 0.0, 1.0)
                     if penalty == 'elasticnet' else None,
            solver='saga',
            max_iter=3000,
            random_state=42,
            class_weight='balanced',
            n_jobs=N_JOBS,
        )

    elif name == 'Ridge':
        base = RidgeClassifier(
            alpha=trial.suggest_float('ridge_alpha', 1e-4, 100, log=True),
            class_weight='balanced',
            random_state=42,
        )
        return CalibratedClassifierCV(base, cv=3, method='sigmoid', n_jobs=N_JOBS)

    elif name == 'LinearSVC':
        penalty = trial.suggest_categorical('lsvc_penalty', ['l1', 'l2'])
        if penalty == 'l1':
            loss = 'squared_hinge'
            dual = False
        else:
            loss = trial.suggest_categorical('lsvc_loss', ['hinge', 'squared_hinge'])
            dual = True if loss == 'hinge' else trial.suggest_categorical('lsvc_dual', [True, False])
        base = LinearSVC(
            C=trial.suggest_float('lsvc_C', 1e-4, 100, log=True),
            penalty=penalty,
            loss=loss,
            dual=dual,
            max_iter=5000,
            random_state=42,
            class_weight='balanced',
        )
        return CalibratedClassifierCV(base, cv=3, method='sigmoid', n_jobs=N_JOBS)

    elif name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_est', 100, 800, step=100),
            max_depth=trial.suggest_int('rf_depth', 3, 30),
            min_samples_split=trial.suggest_int('rf_split', 2, 20),
            min_samples_leaf=trial.suggest_int('rf_leaf', 1, 10),
            max_features=trial.suggest_categorical('rf_feat', ['sqrt', 'log2', None]),
            random_state=42,
            n_jobs=N_JOBS,
            class_weight='balanced',
        )

    elif name == 'ExtraTrees':
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int('et_n_est', 100, 800, step=100),
            max_depth=trial.suggest_int('et_depth', 3, 30),
            min_samples_split=trial.suggest_int('et_split', 2, 20),
            min_samples_leaf=trial.suggest_int('et_leaf', 1, 10),
            max_features=trial.suggest_categorical('et_feat', ['sqrt', 'log2', None]),
            random_state=42,
            n_jobs=N_JOBS,
            class_weight='balanced',
        )

    elif name == 'GBM':
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int('gbm_n_est', 50, 500, step=50),
            max_depth=trial.suggest_int('gbm_depth', 2, 10),
            learning_rate=trial.suggest_float('gbm_lr', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('gbm_sub', 0.5, 1.0),
            min_samples_leaf=trial.suggest_int('gbm_leaf', 1, 15),
            random_state=42,
        )

    elif name == 'XGBoost':
        return xgb.XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_est', 50, 500, step=50),
            max_depth=trial.suggest_int('xgb_depth', 2, 10),
            learning_rate=trial.suggest_float('xgb_lr', 0.005, 0.3, log=True),
            subsample=trial.suggest_float('xgb_sub', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample', 0.3, 1.0),
            min_child_weight=trial.suggest_int('xgb_mcw', 1, 10),
            reg_alpha=trial.suggest_float('xgb_alpha', 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float('xgb_lambda', 1e-8, 10, log=True),
            tree_method='gpu_hist' if HAS_GPU else 'hist',
            device='cuda' if HAS_GPU else 'cpu',
            random_state=42,
            n_jobs=N_JOBS,
            eval_metric='logloss',
            verbosity=0,
        )

    elif name == 'SVM':
        kernel = trial.suggest_categorical('svm_kernel', ['rbf', 'linear', 'poly'])
        return SVC(
            C=trial.suggest_float('svm_C', 1e-3, 100, log=True),
            kernel=kernel,
            gamma=trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
                  if kernel != 'linear' else 'scale',
            degree=trial.suggest_int('svm_degree', 2, 5)
                   if kernel == 'poly' else 3,
            probability=True,
            random_state=42,
            class_weight='balanced',
        )


def rebuild_model(params: dict, n_features: int):
    """Reconstruct model from stored Optuna params."""
    class MockTrial:
        def __init__(self, params):
            self.params = dict(params)
        def suggest_categorical(self, name, choices):
            return self.params[name]
        def suggest_float(self, name, low, high, **kw):
            return self.params[name]
        def suggest_int(self, name, low, high, **kw):
            return self.params[name]

    return suggest_model(MockTrial(params), n_features)


# ─────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────

def create_objective(X_train, y_train, n_features, inner_cv_splits=3):
    def objective(trial):
        model = suggest_model(trial, n_features)
        inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True,
                                   random_state=42)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=inner_cv,
                                     scoring='roc_auc', n_jobs=N_JOBS)
            return scores.mean()
        except Exception:
            return 0.0

    return objective


# ─────────────────────────────────────────────────────────────
# Evaluation: Nested LOGO + Optuna
# ─────────────────────────────────────────────────────────────

def evaluate_nested_logo(X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray,
                         n_trials: int = 50,
                         ) -> Dict[str, object]:
    """
    Nested cross-validation:
      Outer: LOGO (one group held out)
      Inner: Optuna hyperopt with stratified 3-fold on train
    Optimizes AUROC in inner loop.
    """
    logo = LeaveOneGroupOut()
    all_y_true, all_y_prob, all_y_pred = [], [], []
    per_group_auroc = {}
    per_group_model = {}
    per_group_inner_auroc = {}

    n_folds = logo.get_n_splits(X, y, groups)
    n_features = X.shape[1]

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_te = groups[test_idx]
        group_id = g_te[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Inner hyperopt
        objective = create_objective(X_tr_s, y_tr, n_features)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42 + fold_idx),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       n_jobs=1)  # parallelism inside cross_val_score

        # Retrain best on full training fold
        best_model = rebuild_model(study.best_params, n_features)
        best_model.fit(X_tr_s, y_tr)

        y_prob = best_model.predict_proba(X_te_s)[:, 1]
        y_pred = best_model.predict(X_te_s)

        all_y_true.append(y_te)
        all_y_prob.append(y_prob)
        all_y_pred.append(y_pred)

        fold_auroc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else np.nan
        per_group_auroc[group_id] = fold_auroc
        per_group_model[group_id] = study.best_params['model']
        per_group_inner_auroc[group_id] = study.best_value

        print(f"      G{group_id}: "
              f"test_AUROC={fold_auroc:.3f}  "
              f"inner_AUROC={study.best_value:.3f}  "
              f"model={study.best_params['model']}")

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
        'per_group_model': per_group_model,
        'per_group_inner_auroc': per_group_inner_auroc,
    }


# ─────────────────────────────────────────────────────────────
# Feature precomputation
# ─────────────────────────────────────────────────────────────

def precompute_features(df, feature_type, embedding_model):
    features = {}
    if feature_type in ('prosodic', 'all'):
        feats, _ = get_prosodic_features(df)
        features['prosodic'] = feats
    if feature_type in ('transcript', 'all'):
        emb = get_sentence_embeddings(df['transcript'].tolist(),
                                      model_name=embedding_model)
        features['transcript'] = emb
    if feature_type == 'all':
        features['combined'] = np.hstack([features['prosodic'],
                                          features['transcript']])
        print(f"  Combined: {features['combined'].shape[1]}d")
    return features


# ─────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────

def run_experiments(df, feature_type='all', embedding_model='all-mpnet-base-v2',
                    target_states=None, n_seeds=5, n_trials=50):

    print("=" * 70)
    print("Binary One-vs-Rest with Nested Hyperparameter Optimization")
    print(f"  Optimizing: AUROC (inner 3-fold stratified CV)")
    print(f"  Models: {', '.join(MODEL_NAMES)}")
    print(f"  Parallelism: n_jobs={N_JOBS} (all available cores)")
    print("=" * 70)

    all_states = df['label1'].value_counts()
    if target_states is None:
        target_states = all_states.index.tolist()

    print(f"\nTarget states: {target_states}")
    print(f"Resampling seeds: {n_seeds}, Optuna trials/fold: {n_trials}")
    for state in target_states:
        print(f"  {state}: {all_states[state]} positive samples")
    print(f"Unlabeled pool: {df['label1'].isna().sum()}")

    print("\nPrecomputing features...")
    all_features = precompute_features(df, feature_type, embedding_model)

    feat_names = (
        [feature_type] if feature_type in ('prosodic', 'transcript')
        else ['prosodic', 'transcript', 'combined']
    )

    results = []

    for state in target_states:
        print(f"\n{'═' * 70}")
        print(f"State: {state} (n={all_states[state]})")
        print(f"{'═' * 70}")

        for feat_name in feat_names:
            X_full = all_features[feat_name]
            print(f"\n  Feature: {feat_name} ({X_full.shape[1]}d)")

            seed_metrics = []
            seed_per_group = []

            for seed in range(n_seeds):
                print(f"\n    Seed {seed+1}/{n_seeds}")
                rng = np.random.default_rng(seed)
                indices, y, groups = build_binary_dataset(df, state, rng)
                X = X_full[indices]

                metrics = evaluate_nested_logo(X, y, groups, n_trials=n_trials)
                seed_metrics.append(metrics)
                seed_per_group.append(metrics['per_group_auroc'])

                print(f"    → Seed {seed+1} overall AUROC={metrics['auroc']:.3f}")

            # Average across seeds
            scalar_keys = ['auroc', 'f1', 'precision', 'recall']
            avg = {k: np.nanmean([m[k] for m in seed_metrics]) for k in scalar_keys}
            std = {k: np.nanstd([m[k] for m in seed_metrics]) for k in scalar_keys}

            all_group_ids = sorted(set().union(*[pg.keys() for pg in seed_per_group]))
            avg_per_group = {}
            for gid in all_group_ids:
                vals = [pg[gid] for pg in seed_per_group if gid in pg]
                avg_per_group[gid] = np.nanmean(vals)

            model_counts = {}
            for m in seed_metrics:
                for gid, mname in m['per_group_model'].items():
                    model_counts[mname] = model_counts.get(mname, 0) + 1

            result_row = {
                'state': state,
                'n_positive': all_states[state],
                'feature_type': feat_name,
                'auroc_mean': avg['auroc'],
                'auroc_std': std['auroc'],
                'f1_mean': avg['f1'],
                'f1_std': std['f1'],
                'precision_mean': avg['precision'],
                'precision_std': std['precision'],
                'recall_mean': avg['recall'],
                'recall_std': std['recall'],
                'model_selection': str(model_counts),
            }
            for gid in all_group_ids:
                result_row[f'auroc_group_{gid}'] = avg_per_group[gid]

            results.append(result_row)

            gstr = "  ".join(
                f"G{gid}={avg_per_group[gid]:.2f}"
                if not np.isnan(avg_per_group[gid]) else f"G{gid}=n/a"
                for gid in all_group_ids
            )
            print(f"\n  ╔═ {state} / {feat_name}: "
                  f"AUROC={avg['auroc']:.3f}±{std['auroc']:.3f}  "
                  f"F1={avg['f1']:.3f}±{std['f1']:.3f}")
            print(f"  ║  Per-group: {gstr}")
            print(f"  ╚  Models: {model_counts}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Binary affect classification with nested hyperopt"
    )
    parser.add_argument("input_file", help="Path to preprocessed CSV")
    parser.add_argument("-f", "--features",
                        choices=['prosodic', 'transcript', 'all'], default='all')
    parser.add_argument("--states", nargs='+', default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials per LOGO fold (default: 50)")
    parser.add_argument("--embedding-model", default='all-mpnet-base-v2')
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    print(f"Loading: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples "
          f"({df['label1'].notna().sum()} labeled, "
          f"{df['label1'].isna().sum()} unlabeled)")

    results_df = run_experiments(
        df,
        feature_type=args.features,
        embedding_model=args.embedding_model,
        target_states=args.states,
        n_seeds=args.seeds,
        n_trials=args.trials,
    )

    if args.output is None:
        p = Path(args.input_file)
        output_path = p.parent / f"hyperopt_results_{args.features}.csv"
    else:
        output_path = args.output

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for state in results_df['state'].unique():
        sub = results_df[results_df['state'] == state]
        best = sub.loc[sub['auroc_mean'].idxmax()]
        print(f"  {state:12s}: AUROC={best['auroc_mean']:.3f}±{best['auroc_std']:.3f}  "
              f"F1={best['f1_mean']:.3f}  ({best['feature_type']})  "
              f"{best['model_selection']}")


if __name__ == "__main__":
    main()