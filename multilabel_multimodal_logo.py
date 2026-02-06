"""
Multi-Label Classification with Three Modalities: Prosody, Text, and Combined

Experiments:
1. Prosody-only (88 acoustic features)
2. Text-only (TF-IDF from transcripts)
3. Combined (prosody + text)

All with Leave-One-Group-Out CV and hyperopt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score, precision_score, recall_score,
    jaccard_score, classification_report, roc_auc_score, cohen_kappa_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# BERT imports
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

# For hyperparameter optimization
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: hyperopt not available. Install with: pip install hyperopt")

RESULTS_DIR = '/home/videep/research/predicting-affectivestates/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


class MultiModalMultiLabelModeling:
    """Multi-label classification with text, prosody, or combined features"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_multilabel = None
        self.prosodic_features = None
        self.mlb = MultiLabelBinarizer()
        self.bert_model = None
        self.bert_tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if BERT_AVAILABLE else None
        
    def load_and_prepare(self):
        """Load and prepare data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        self.df = pd.read_excel(self.data_path)
        
        # Identify prosodic features
        exclude_cols = ['timestamp', 'videoTime', 'labels', 'self-caught', 'probe-caught',
                       'groupID', 'participantID', 'participant', 'onset', 'ongoing', 
                       'duration', 'videoEndTime', 'Other', 'TimeDifference', 'phase_type',
                       'caught_type', 'performance_type', 'startTime', 'endTime', 
                       'windowLength', 'transcript']
        
        self.prosodic_features = [col for col in self.df.columns 
                                 if col not in exclude_cols and 
                                 self.df[col].dtype in ['float64', 'int64']]
        
        print(f"Total rows: {len(self.df)}")
        print(f"Identified {len(self.prosodic_features)} prosodic features")
        
        # Check transcript availability
        transcripts_available = self.df['transcript'].notna().sum()
        print(f"Rows with transcripts: {transcripts_available}")
        
        return self
    
    def create_multilabel_dataset(self, min_samples_per_class=5):
        """
        Create multi-label dataset by aggregating rows with same time window
        """
        print("\n" + "="*80)
        print("CREATING MULTI-LABEL DATASET")
        print("="*80)
        
        # Remove rows with missing labels
        df_clean = self.df.dropna(subset=['labels']).copy()
        print(f"Rows with labels: {len(df_clean)}")
        
        # Filter rare classes
        label_counts = df_clean['labels'].value_counts()
        valid_labels = label_counts[label_counts >= min_samples_per_class].index.tolist()
        df_clean = df_clean[df_clean['labels'].isin(valid_labels)]
        print(f"After filtering rare classes: {len(df_clean)} rows")
        print(f"Valid labels: {valid_labels}")
        
        # Create unique identifier for each time window
        df_clean['window_id'] = (
            df_clean['participantID'].astype(str) + '_' + 
            df_clean['startTime'].astype(str) + '_' + 
            df_clean['endTime'].astype(str)
        )
        
        # Fill missing prosodic values with mean
        for col in self.prosodic_features:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        # Fill missing transcripts with empty string
        df_clean['transcript'].fillna('', inplace=True)
        
        # Aggregate by window_id
        print("\nAggregating labels by time window...")
        
        agg_dict = {
            'labels': lambda x: list(set(x)),  # Unique labels
            'participantID': 'first',
            'participant': 'first',
            'groupID': 'first',
            'transcript': 'first'  # Take first transcript (should be same)
        }
        
        # Add prosodic features (take mean if multiple rows)
        for col in self.prosodic_features:
            agg_dict[col] = 'mean'
        
        df_multilabel = df_clean.groupby('window_id').agg(agg_dict).reset_index()
        
        print(f"\nUnique time windows: {len(df_multilabel)}")
        print(f"Original rows: {len(df_clean)}")
        print(f"Reduction: {len(df_clean) - len(df_multilabel)} rows aggregated")
        
        # Analyze multi-label statistics
        label_counts_per_window = df_multilabel['labels'].apply(len)
        print(f"\nLabels per window:")
        print(f"  Mean: {label_counts_per_window.mean():.2f}")
        print(f"  Median: {label_counts_per_window.median():.0f}")
        print(f"  Max: {label_counts_per_window.max():.0f}")
        print(f"  Windows with >1 label: {(label_counts_per_window > 1).sum()} "
              f"({(label_counts_per_window > 1).sum() / len(df_multilabel) * 100:.1f}%)")
        
        # Show label distribution
        print("\nLabel distribution:")
        all_labels = [label for labels in df_multilabel['labels'] for label in labels]
        label_dist = pd.Series(all_labels).value_counts()
        print(label_dist)
        
        # Check text availability
        text_available = (df_multilabel['transcript'] != '').sum()
        print(f"\nWindows with transcripts: {text_available} ({text_available/len(df_multilabel)*100:.1f}%)")
        
        self.df_multilabel = df_multilabel
        return df_multilabel
    
    def get_bert_embeddings(self, texts, model_name='bert-base-uncased', batch_size=16):
        """
        Extract BERT embeddings for a list of texts
        
        Args:
            texts: List of text strings
            model_name: Name of the pre-trained BERT model
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings (n_samples, 768)
        """
        if not BERT_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers torch")
        
        print(f"\nLoading BERT model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        if self.bert_tokenizer is None:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
        
        embeddings = []
        
        # Process in batches
        print(f"Extracting BERT embeddings for {len(texts)} texts...")
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {key: val.to(self.device) for key, val in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**encoded)
                # Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        print(f"BERT embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def prepare_features(self, feature_type='prosody', bert_model='bert-base-uncased'):
        """
        Prepare features based on modality
        
        Args:
            feature_type: 'prosody', 'text', or 'combined'
            bert_model: BERT model name (e.g., 'bert-base-uncased', 'distilbert-base-uncased')
            
        Returns:
            X: Feature matrix
            y: Multi-label targets
            groups: Group IDs for CV
        """
        print(f"\n" + "="*80)
        print(f"PREPARING FEATURES: {feature_type.upper()}")
        print("="*80)
        
        # Multi-label targets
        y = self.mlb.fit_transform(self.df_multilabel['labels'])
        
        # Groups for LOGO CV
        groups = self.df_multilabel['groupID'].values
        
        if feature_type == 'prosody':
            # Prosodic features only
            X = self.df_multilabel[self.prosodic_features].values
            print(f"Using {X.shape[1]} prosodic features")
            
        elif feature_type == 'text':
            # Text features only (BERT embeddings)
            transcripts = self.df_multilabel['transcript'].fillna('').tolist()
            
            # Filter out samples with no transcript
            has_text = [t != '' for t in transcripts]
            if sum(has_text) < len(transcripts):
                print(f"Warning: {len(transcripts) - sum(has_text)} samples have no transcript")
                print("Filling empty transcripts with placeholder")
                transcripts = ['no transcript available' if t == '' else t for t in transcripts]
            
            # Get BERT embeddings
            X = self.get_bert_embeddings(transcripts, model_name=bert_model)
            print(f"Using {X.shape[1]} BERT embedding features")
            
        elif feature_type == 'combined':
            # Combined: prosody + text (BERT)
            X_prosody = self.df_multilabel[self.prosodic_features].values
            
            transcripts = self.df_multilabel['transcript'].fillna('').tolist()
            has_text = [t != '' for t in transcripts]
            if sum(has_text) < len(transcripts):
                transcripts = ['no transcript available' if t == '' else t for t in transcripts]
            
            # Get BERT embeddings
            X_text = self.get_bert_embeddings(transcripts, model_name=bert_model)
            
            # Concatenate features
            X = np.hstack([X_prosody, X_text])
            print(f"Using {X_prosody.shape[1]} prosodic + {X_text.shape[1]} BERT = {X.shape[1]} total features")
            
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        print(f"\nData prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of groups: {len(np.unique(groups))}")
        print(f"  Label classes: {self.mlb.classes_}")
        
        return X, y, groups
    
    def evaluate_multilabel_model(self, y_true, y_pred, y_pred_proba=None):
        """Comprehensive multi-label evaluation"""
        results = {
            'hamming_loss': hamming_loss(y_true, y_pred),
            'subset_accuracy': accuracy_score(y_true, y_pred),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_samples': f1_score(y_true, y_pred, average='samples'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'jaccard_macro': jaccard_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class F1 scores
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        results['per_class_f1'] = dict(zip(self.mlb.classes_, per_class_f1))
        
        # Per-class Kappa
        per_class_kappa = []
        for i in range(y_true.shape[1]):
            try:
                kappa = cohen_kappa_score(y_true[:, i], y_pred[:, i])
                per_class_kappa.append(kappa)
            except:
                per_class_kappa.append(0.0)
        results['per_class_kappa'] = dict(zip(self.mlb.classes_, per_class_kappa))
        results['kappa_macro'] = np.mean(per_class_kappa)
        
        # Per-class AUROC (if probabilities provided)
        if y_pred_proba is not None:
            per_class_auroc = []
            for i in range(y_true.shape[1]):
                try:
                    if len(np.unique(y_true[:, i])) > 1:
                        auroc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                    else:
                        auroc = 0.5
                    per_class_auroc.append(auroc)
                except:
                    per_class_auroc.append(0.5)
            results['per_class_auroc'] = dict(zip(self.mlb.classes_, per_class_auroc))
            results['auroc_macro'] = np.mean(per_class_auroc)
        
        return results
    
    def train_with_logo_cv(self, X, y, groups, model_params=None, feature_type='prosody'):
        """Train with Leave-One-Group-Out cross-validation"""
        print("\n" + "="*80)
        print(f"LEAVE-ONE-GROUP-OUT CV - {feature_type.upper()}")
        print("="*80)
        
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(X, y, groups)
        
        print(f"Number of folds: {n_splits} (one per group)")
        print(f"Model params: {model_params}")
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            test_group = groups[test_idx[0]]
            print(f"\nFold {fold_idx+1}/{n_splits} - Testing group: {test_group}")
            print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            base_model = RandomForestClassifier(**model_params)
            model = MultiOutputClassifier(base_model)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Get probabilities for AUROC
            y_pred_proba = np.zeros((len(X_test_scaled), y_test.shape[1]))
            for i, estimator in enumerate(model.estimators_):
                proba = estimator.predict_proba(X_test_scaled)
                y_pred_proba[:, i] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            
            # Evaluate
            fold_result = self.evaluate_multilabel_model(y_test, y_pred, y_pred_proba)
            fold_result['fold'] = fold_idx + 1
            fold_result['group'] = test_group
            fold_results.append(fold_result)
            
            # Store for overall metrics
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_pred_proba.append(y_pred_proba)
            
            # Print fold results
            print(f"  F1 (samples): {fold_result['f1_samples']:.4f}")
            print(f"  AUROC (macro): {fold_result['auroc_macro']:.4f}")
        
        # Aggregate results
        all_y_true = np.vstack(all_y_true)
        all_y_pred = np.vstack(all_y_pred)
        all_y_pred_proba = np.vstack(all_y_pred_proba)
        overall_results = self.evaluate_multilabel_model(all_y_true, all_y_pred, all_y_pred_proba)
        
        # Compute statistics across folds
        fold_df = pd.DataFrame(fold_results)
        
        print("\n" + "="*80)
        print("CROSS-VALIDATION RESULTS")
        print("="*80)
        
        metrics = ['hamming_loss', 'subset_accuracy', 'f1_micro', 'f1_macro', 
                  'f1_weighted', 'f1_samples', 'jaccard_macro', 'auroc_macro', 'kappa_macro']
        
        print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-"*60)
        for metric in metrics:
            if metric in fold_df.columns:
                values = fold_df[metric].values
                print(f"{metric:<20} {values.mean():.4f}    {values.std():.4f}    "
                      f"{values.min():.4f}    {values.max():.4f}")
        
        print("\nOverall (all folds combined):")
        for metric in metrics:
            if metric in overall_results:
                print(f"  {metric:<20}: {overall_results[metric]:.4f}")
        
        # Print per-label metrics
        print("\n" + "="*80)
        print("PER-LABEL METRICS (Overall)")
        print("="*80)
        print(f"\n{'Label':<15} {'F1':<10} {'AUROC':<10} {'Kappa':<10}")
        print("-"*45)
        for label in self.mlb.classes_:
            f1 = overall_results['per_class_f1'][label]
            auroc = overall_results['per_class_auroc'][label] if 'per_class_auroc' in overall_results else 0.0
            kappa = overall_results['per_class_kappa'][label]
            print(f"{label:<15} {f1:<10.4f} {auroc:<10.4f} {kappa:<10.4f}")
        
        return fold_results, overall_results, fold_df


def main():
    """Main execution - run all three modalities"""
    
    print("="*80)
    print("MULTI-MODAL MULTI-LABEL AFFECTIVE STATE PREDICTION")
    print("Testing: Prosody, Text, and Combined")
    print("="*80)
    
    # Initialize
    data_path = '/home/videep/research/predicting-affectivestates/reports_agg_20s_windows_transcripts_opensmile_egemaps1.xlsx'
    modeler = MultiModalMultiLabelModeling(data_path)
    
    # Load and prepare
    modeler.load_and_prepare()
    modeler.create_multilabel_dataset(min_samples_per_class=5)
    
    # Default model parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    results_summary = []
    
    # EXPERIMENT 1: PROSODY ONLY
    print("\n" + "="*80)
    print("EXPERIMENT 1: PROSODY ONLY")
    print("="*80)
    
    X_prosody, y, groups = modeler.prepare_features(feature_type='prosody')
    fold_results_prosody, overall_prosody, fold_df_prosody = modeler.train_with_logo_cv(
        X_prosody, y, groups, model_params=default_params, feature_type='prosody'
    )
    
    results_summary.append({
        'Modality': 'Prosody Only',
        'Features': X_prosody.shape[1],
        'F1 (samples)': f"{fold_df_prosody['f1_samples'].mean():.4f} ± {fold_df_prosody['f1_samples'].std():.4f}",
        'F1 (macro)': f"{fold_df_prosody['f1_macro'].mean():.4f} ± {fold_df_prosody['f1_macro'].std():.4f}",
        'AUROC': f"{fold_df_prosody['auroc_macro'].mean():.4f} ± {fold_df_prosody['auroc_macro'].std():.4f}",
        'Kappa': f"{fold_df_prosody['kappa_macro'].mean():.4f} ± {fold_df_prosody['kappa_macro'].std():.4f}",
        'Hamming Loss': f"{fold_df_prosody['hamming_loss'].mean():.4f} ± {fold_df_prosody['hamming_loss'].std():.4f}"
    })
    
    # EXPERIMENT 2: TEXT ONLY
    print("\n" + "="*80)
    print("EXPERIMENT 2: TEXT ONLY (BERT EMBEDDINGS)")
    print("="*80)
    
    X_text, y, groups = modeler.prepare_features(feature_type='text', bert_model='bert-base-uncased')
    fold_results_text, overall_text, fold_df_text = modeler.train_with_logo_cv(
        X_text, y, groups, model_params=default_params, feature_type='text'
    )
    
    results_summary.append({
        'Modality': 'Text Only',
        'Features': X_text.shape[1],
        'F1 (samples)': f"{fold_df_text['f1_samples'].mean():.4f} ± {fold_df_text['f1_samples'].std():.4f}",
        'F1 (macro)': f"{fold_df_text['f1_macro'].mean():.4f} ± {fold_df_text['f1_macro'].std():.4f}",
        'AUROC': f"{fold_df_text['auroc_macro'].mean():.4f} ± {fold_df_text['auroc_macro'].std():.4f}",
        'Kappa': f"{fold_df_text['kappa_macro'].mean():.4f} ± {fold_df_text['kappa_macro'].std():.4f}",
        'Hamming Loss': f"{fold_df_text['hamming_loss'].mean():.4f} ± {fold_df_text['hamming_loss'].std():.4f}"
    })
    
    # EXPERIMENT 3: COMBINED
    print("\n" + "="*80)
    print("EXPERIMENT 3: PROSODY + BERT COMBINED")
    print("="*80)
    
    X_combined, y, groups = modeler.prepare_features(feature_type='combined', bert_model='bert-base-uncased')
    fold_results_combined, overall_combined, fold_df_combined = modeler.train_with_logo_cv(
        X_combined, y, groups, model_params=default_params, feature_type='combined'
    )
    
    results_summary.append({
        'Modality': 'Prosody + Text',
        'Features': X_combined.shape[1],
        'F1 (samples)': f"{fold_df_combined['f1_samples'].mean():.4f} ± {fold_df_combined['f1_samples'].std():.4f}",
        'F1 (macro)': f"{fold_df_combined['f1_macro'].mean():.4f} ± {fold_df_combined['f1_macro'].std():.4f}",
        'AUROC': f"{fold_df_combined['auroc_macro'].mean():.4f} ± {fold_df_combined['auroc_macro'].std():.4f}",
        'Kappa': f"{fold_df_combined['kappa_macro'].mean():.4f} ± {fold_df_combined['kappa_macro'].std():.4f}",
        'Hamming Loss': f"{fold_df_combined['hamming_loss'].mean():.4f} ± {fold_df_combined['hamming_loss'].std():.4f}"
    })
    
    # FINAL COMPARISON
    print("\n" + "="*80)
    print("FINAL COMPARISON: ALL MODALITIES")
    print("="*80)
    
    comparison_df = pd.DataFrame(results_summary)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, f'multimodal_comparison_{timestamp}.csv'),
        index=False
    )
    
    # Save detailed results
    all_results = {
        'prosody': {
            'fold_results': fold_df_prosody.to_dict(),
            'overall': overall_prosody,
        },
        'text': {
            'fold_results': fold_df_text.to_dict(),
            'overall': overall_text,
        },
        'combined': {
            'fold_results': fold_df_combined.to_dict(),
            'overall': overall_combined,
        },
        'comparison': comparison_df.to_dict(),
        'label_classes': modeler.mlb.classes_.tolist()
    }
    
    joblib.dump(
        all_results,
        os.path.join(RESULTS_DIR, f'multimodal_results_{timestamp}.pkl')
    )
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - multimodal_comparison_{timestamp}.csv")
    print(f"  - multimodal_results_{timestamp}.pkl")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return modeler, all_results


if __name__ == "__main__":
    modeler, results = main()
