# Affective State Prediction from Dialogue Transcripts and Prosody

Multi-label classification of affective states using BERT embeddings and prosodic features with Leave-One-Group-Out cross-validation.

## Overview

This repository contains code for predicting 7 affective states from educational dialogue:
- **Optimistic**
- **Curious** 
- **Confused**
- **Conflicted**
- **Surprised**
- **Disengaged**
- **Frustrated**

The analysis compares:
- **3 time window sizes**: 5s, 10s, 20s
- **3 modalities**: Prosody-only, Text-only (BERT), Combined
- **Evaluation**: Leave-One-Group-Out CV (9 groups)

## Key Results

### Best Overall Performance
- **Best AUROC**: 0.581 (10s window, Text-only)
- **Best F1**: 0.071 (20s window, Prosody-only)

### Emotions Achieving AUROC ≥ 0.6
1. **Disengaged**: 0.679 (10s, Text)
2. **Conflicted**: 0.658 (5s, Combined)
3. **Surprised**: 0.634 (20s, Combined)
4. **Curious**: 0.632 (10s, Text)
5. **Optimistic**: 0.612 (5s, Prosody)

**5 out of 7 emotions are highly predictable!**

## Repository Structure

```
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── Data Files
│   ├── reports_agg_5s_windows_transcripts.csv
│   ├── reports_agg_10s_windows_transcripts.csv
│   └── reports_agg_20s_windows_transcripts_opensmile_egemaps1.xlsx
│
├── Analysis Scripts
│   ├── multilabel_with_none_class.py      # Main analysis script (filtered emotions)
│   ├── multilabel_window_comparison.py    # Window size comparison (all emotions)
│   ├── multilabel_multimodal_logo.py      # Original multimodal analysis
│   ├── compare_window_sizes.py            # Cross-window comparison & visualization
│   └── display_bert_results.py            # BERT vs TF-IDF comparison
│
├── Results
│   ├── with_none_class_*_comparison_*.csv # Filtered emotion results
│   ├── window_*_comparison_*.csv          # Window comparison results
│   ├── multimodal_comparison_*.csv        # Original multimodal results
│   └── *.png                              # Visualizations
│
└── Documentation
    ├── FILTERED_EMOTIONS_ANALYSIS.md      # Analysis of 7 filtered emotions
    ├── WINDOW_SIZE_ANALYSIS.md            # Window size comparison
    ├── BERT_VS_TFIDF.md                   # Text feature comparison
    ├── MULTIMODAL_COMPARISON.md           # TF-IDF multimodal analysis
    └── MULTILABEL_LOGO_RESULTS.md         # Initial prosody-only results
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd predicting-affectivestates
```

### 2. Create Conda Environment
```bash
conda create -n basic_ml python=3.11
conda activate basic_ml
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl (for Excel files)
- transformers
- torch (with CUDA support recommended)
- tqdm

### 4. Verify CUDA (Optional but Recommended)
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Usage

### Quick Start: Run Full Analysis on 10s Windows

```bash
conda activate basic_ml
python multilabel_with_none_class.py reports_agg_10s_windows_transcripts.csv 10s
```

This will:
1. Load and filter data (7 target emotions)
2. Extract BERT embeddings (768-dim)
3. Train models with LOGO CV (9 folds)
4. Save results to `results/with_none_class_10s_comparison_*.csv`

Expected runtime: ~6-8 minutes (with GPU)

### Run on All Window Sizes

```bash
# 5-second windows
python multilabel_with_none_class.py reports_agg_5s_windows_transcripts.csv 5s

# 10-second windows  
python multilabel_with_none_class.py reports_agg_10s_windows_transcripts.csv 10s

# 20-second windows
python multilabel_with_none_class.py reports_agg_20s_windows_transcripts_opensmile_egemaps1.xlsx 20s
```

### Generate Comparison Visualizations

```bash
# Compare window sizes
python compare_window_sizes.py

# Compare BERT vs TF-IDF (for 20s windows)
python display_bert_results.py
```

## Data Format

### Input CSV/Excel Requirements

The input data should have these columns:

**Required:**
- `labels`: Comma-separated emotion labels (e.g., "Optimistic, Curious")
- `transcript`: Text transcript of the dialogue window
- `groupID`: Group identifier for LOGO CV
- `participantID`: Participant identifier
- `timestamp` or `videoTime`: Time identifier for window aggregation

**Prosodic Features (88 eGeMAPS):**
- F0 features (pitch)
- Jitter, shimmer (voice quality)
- HNR (harmonics-to-noise ratio)
- Loudness
- Spectral features
- MFCCs
- Alpha ratio, Hammarberg index, spectral slope

### Output Results

Each run generates a CSV with:
- `WindowSize`: 5s, 10s, or 20s
- `Modality`: prosody, text, or combined
- `F1_samples_mean/std`: F1 score (samples averaging)
- `AUROC_macro_mean/std`: Macro-averaged AUROC
- `Kappa_macro_mean/std`: Cohen's Kappa
- `Hamming_loss_mean/std`: Hamming loss
- `AUROC_{emotion}_mean/std`: Per-emotion AUROC for each label

## Key Scripts Explained

### multilabel_with_none_class.py

**Main analysis script for filtered emotions.**

Features:
- Filters to 7 target emotions only
- Removes rows with invalid/rare labels
- Multi-label classification (windows can have multiple emotions)
- BERT embeddings (bert-base-uncased)
- Leave-One-Group-Out CV
- Per-label AUROC and Kappa metrics

Usage:
```bash
python multilabel_with_none_class.py <input_file> <window_size>
```

### multilabel_window_comparison.py

**Compares different window sizes with all emotion labels.**

Identical to multilabel_with_none_class.py but doesn't filter emotions.

### compare_window_sizes.py

**Generates comprehensive comparison visualizations.**

Creates:
- Bar charts comparing F1 and AUROC across window sizes
- Heatmaps of per-emotion performance
- `window_size_comparison.png`
- `results/window_size_comprehensive_comparison.csv`

### display_bert_results.py

**Compares BERT vs TF-IDF text features.**

Analyzes trade-offs between simpler (TF-IDF) and complex (BERT) text representations.

## Methodology

### Data Preprocessing

1. **Load data** from CSV/Excel
2. **Filter emotions** to 7 target labels
3. **Remove invalid rows**:
   - Missing prosodic features
   - Missing transcripts
   - Labels not in target set
4. **Aggregate by window**: Combine multiple rows for same time window
5. **Multi-label encoding**: Binary vector for each emotion

### Feature Extraction

**Prosody (88 features):**
- OpenSMILE eGeMAPS feature set
- Standardized with StandardScaler

**Text (768 features):**
- BERT-base-uncased model
- [CLS] token embeddings
- Batch processing (16 samples/batch)
- GPU acceleration

**Combined (856 features):**
- Concatenation of prosody + BERT
- Note: Simple concatenation underperforms vs single modalities

### Model Training

**Algorithm**: RandomForestClassifier with MultiOutputClassifier
- 100 trees
- Max depth: 10
- Min samples split: 5
- Min samples leaf: 2

**Cross-Validation**: Leave-One-Group-Out (LOGO)
- 9 folds (one per group)
- Prevents data leakage between groups
- Tests generalization to new groups

**Metrics**:
- F1 score (samples, macro, micro, weighted)
- AUROC per label (handles class imbalance)
- Cohen's Kappa per label
- Hamming loss
- Subset accuracy

## Results Summary

See detailed analysis in:
- [FILTERED_EMOTIONS_ANALYSIS.md](FILTERED_EMOTIONS_ANALYSIS.md) - Main results
- [WINDOW_SIZE_ANALYSIS.md](WINDOW_SIZE_ANALYSIS.md) - Window comparison
- [BERT_VS_TFIDF.md](BERT_VS_TFIDF.md) - Text feature comparison

### Key Findings

1. **Text (BERT) achieves best discrimination** (AUROC ≈ 0.58)
2. **Prosody achieves best prediction accuracy** (F1 ≈ 0.07)
3. **10s windows are optimal** for text-based models
4. **5 out of 7 emotions** achieve AUROC ≥ 0.6
5. **Combined features underperform** with simple concatenation

### Emotion-Specific Recommendations

| Emotion | Best Config | AUROC | Strategy |
|---------|-------------|-------|----------|
| Disengaged | 10s, Text | 0.679 | Text-only, semantic cues |
| Conflicted | 5s, Combined | 0.658 | Fusion, rapid detection |
| Surprised | 20s, Combined | 0.634 | Fusion, longer context |
| Curious | 10s, Text | 0.632 | Text-only, semantic |
| Optimistic | 5s, Prosody | 0.612 | Prosody-only, vocal cues |
| Confused | 20s, Combined | 0.599 | Fusion, longer context |
| Frustrated | 5s, Combined | 0.579 | Fusion, rapid detection |

## Replication

To fully replicate the published results:

1. **Setup environment** (see Installation)
2. **Run all window sizes**:
   ```bash
   python multilabel_with_none_class.py reports_agg_5s_windows_transcripts.csv 5s
   python multilabel_with_none_class.py reports_agg_10s_windows_transcripts.csv 10s
   python multilabel_with_none_class.py reports_agg_20s_windows_transcripts_opensmile_egemaps1.xlsx 20s
   ```
3. **Generate visualizations**:
   ```bash
   python compare_window_sizes.py
   ```
4. **Results** will be in `results/` directory

### Expected Runtime

- 5s windows: ~9 minutes (516 windows)
- 10s windows: ~6 minutes (311 windows)
- 20s windows: ~5 minutes (289 windows)

GPU recommended but not required (CPU is ~3x slower).

### Random Seed

Models use `random_state=42` for reproducibility. Results may vary slightly due to:
- BERT model version updates
- Floating-point precision
- CUDA non-determinism

## Troubleshooting

### Out of Memory (CUDA)

Reduce batch size in BERT extraction:
```python
batch_size=8  # Default is 16
```

### Missing CUDA

Models will automatically fall back to CPU. Add `device='cpu'` explicitly if needed.

### Missing Dependencies

```bash
pip install --upgrade transformers torch
```

### Column Name Issues

If timestamp column not found, script will try alternatives:
- `timestamp`
- `¡timestamp` (encoding issues)
- `videoTime`

## Citation

If you use this code, please cite:

```
@misc{affective-state-prediction-2026,
  title={Multi-label Affective State Prediction from Educational Dialogue},
  author={},
  year={2026},
  note={7 emotions with BERT + prosody, LOGO CV}
}
```

## License

[Specify your license]

## Contact

[Your contact information]

## Acknowledgments

- BERT model: Hugging Face Transformers
- Prosodic features: OpenSMILE eGeMAPS
- Cross-validation approach: Leave-One-Group-Out

---

*Last Updated: February 6, 2026*
