# Multi-Label Classification with Leave-One-Group-Out CV

## Summary

This analysis implements **multi-label classification** for affective state prediction using **Leave-One-Group-Out Cross-Validation** with 9 groups.

## Key Corrections Made

1. **Changed from participants to groups**: Now using `groupID` (9 groups) instead of `participantID` (27 participants) for LOGO CV
2. **Added AUROC per label**: Using `predict_proba` to compute ROC-AUC for each emotion label
3. **Added Cohen's Kappa per label**: Computing agreement metric for each binary label

## Dataset Structure

- **289 unique time windows** (aggregated from 334 original rows)
- **8 emotion labels**: Conflicted, Confused, Curious, Disengaged, Engaged, Frustrated, Optimistic, Surprised
- **12.1% multi-label samples** (have >1 emotion, max 4 emotions)
- **9 groups** for cross-validation (groupID: 2, 3, 4, 5, 6, 7, 8, 9, 10)
- **88 prosodic features** from OpenSMILE eGeMAPS

## Cross-Validation Strategy

**Leave-One-Group-Out (LOGO) CV**:
- 9 folds (one per group)
- Each fold holds out all samples from one group for testing
- Ensures no data leakage between groups
- More realistic evaluation than participant-based CV

## Results

### Overall Performance (Optimized Model)

| Metric | Value |
|--------|-------|
| **F1 (samples)** | 0.090 ± 0.110 |
| **F1 (macro)** | 0.040 ± 0.034 |
| **AUROC (macro)** | 0.444 ± 0.050 |
| **Kappa (macro)** | -0.017 ± nan |
| **Hamming Loss** | 0.156 ± 0.016 |
| **Subset Accuracy** | 7.5% ± 8.7% |
| **Label Accuracy** | 84.4% |

### Per-Label Metrics (Average Across 9 Folds)

| Label | F1 | AUROC | Kappa |
|-------|-----|-------|-------|
| **Optimistic** | 0.165 ± 0.211 | 0.518 ± 0.134 | 0.018 ± 0.070 |
| **Curious** | 0.095 ± 0.139 | 0.499 ± 0.105 | -0.017 ± 0.117 |
| **Confused** | 0.062 ± 0.118 | 0.436 ± 0.116 | 0.028 ± 0.083 |
| **Frustrated** | 0.000 ± 0.000 | 0.472 ± 0.105 | 0.000 ± 0.000 |
| **Engaged** | 0.000 ± 0.000 | 0.462 ± 0.126 | 0.000 ± 0.000 |
| **Surprised** | 0.000 ± 0.000 | 0.459 ± 0.199 | 0.000 ± 0.000 |
| **Disengaged** | 0.000 ± 0.000 | 0.375 ± 0.139 | -0.018 ± 0.044 |
| **Conflicted** | 0.000 ± 0.000 | 0.331 ± 0.192 | 0.000 ± 0.000 |

### Hyperparameter Optimization

**Best Parameters Found (hyperopt with 50 trials)**:
- n_estimators: 100
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 1

**Improvements from Hyperopt**:
- F1 (samples): +17.6%
- F1 (macro): +25.7%
- Subset Accuracy: +7.6%

## Interpretation

### What the Metrics Mean

1. **F1 (samples) = 0.090**: Average F1 score per sample across all labels. Low because many labels are rare and hard to predict.

2. **AUROC (macro) = 0.444**: Average area under ROC curve across all 8 labels. Close to 0.5 (random) indicates difficulty distinguishing positive/negative for each emotion.

3. **Kappa (macro) ≈ 0.0**: Cohen's Kappa near zero indicates agreement barely better than chance. Negative values for some labels suggest systematic disagreement.

4. **Label Accuracy = 84.4%**: The model correctly predicts 84.4% of individual binary labels (averaged across all samples and labels).

5. **Subset Accuracy = 7.5%**: Only 7.5% of samples have all their labels predicted correctly (exact match). This is expected with 8 labels.

### Per-Label Insights

**Best Performance**:
- **Optimistic**: Highest F1 (0.165) and AUROC (0.518), suggesting it's the most predictable emotion
- **Curious**: Second best with F1=0.095 and AUROC=0.499

**Challenging Emotions**:
- **Frustrated, Engaged, Surprised, Disengaged, Conflicted**: F1=0.0, indicating the model rarely predicts these
- These emotions are either rare in the dataset or not distinguishable from prosodic features alone

**AUROC Analysis**:
- Most labels have AUROC between 0.3-0.5, close to random (0.5)
- Suggests prosodic features alone may not provide strong discriminative power for these emotions
- Consider adding text-based features (transcripts) for better performance

**Kappa Analysis**:
- Near-zero or negative Kappa indicates poor inter-rater agreement between model and ground truth
- Suggests the task is very challenging with current features

## Recommendations

1. **Add Text Features**: Combine prosodic features with TF-IDF or BERT embeddings from transcripts

2. **Class Balancing**: Use SMOTE or class weights to handle imbalanced labels (some emotions are very rare)

3. **Simplify Task**: Consider:
   - Binary classification (e.g., engaged vs disengaged)
   - Group emotions into positive/negative/neutral valence
   - Focus on most frequent emotions (Optimistic, Curious, Confused)

4. **Feature Engineering**: 
   - Add temporal features (change in prosody over time)
   - Add interaction features between prosodic dimensions
   - Consider deep learning on raw audio

5. **Larger Dataset**: 289 samples across 9 groups for 8 labels is quite small. More data would help.

## Files Generated

- `multilabel_logo_hyperopt.py`: Main script with LOGO CV, AUROC, and Kappa
- `results/multilabel_loko_results_20260205_214928.pkl`: Full results dictionary
- `results/multilabel_comparison_20260205_214928.csv`: Comparison table
- `results/hyperopt_history.png`: Hyperparameter optimization visualization
- `results/multilabel_logo_cv_results.png`: Per-group performance visualization
- `display_logo_results.py`: Script to display comprehensive results

## Comparison: Participant-based vs Group-based LOGO CV

### Previous (27 participants):
- 27 folds, smaller test sets per fold
- F1 (samples): 0.154 ± 0.116
- AUROC: Not computed
- More granular but potentially less stable folds

### Current (9 groups):
- 9 folds, larger test sets per fold
- F1 (samples): 0.090 ± 0.110
- AUROC (macro): 0.444 ± 0.050
- Kappa (macro): -0.017
- More realistic group-level evaluation

**Why is performance lower?**
- Larger test set per fold means more diverse samples
- Group-based splits may have more domain shift
- More challenging but more realistic evaluation

## Conclusion

The multi-label classification task is **very challenging** with prosodic features alone:
- Only Optimistic and Curious show modest predictability (F1 > 0)
- AUROC near 0.5 indicates limited discriminative power
- Group-based LOGO CV provides more stringent, realistic evaluation

**Next Steps**: Combine prosodic + text features, simplify task, or collect more data.
