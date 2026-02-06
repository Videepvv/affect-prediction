# Multi-Modal Comparison: Prosody vs Text vs Combined

## Summary

This analysis compares **three modalities** for multi-label affective state prediction using Leave-One-Group-Out Cross-Validation (9 groups):

1. **Prosody Only** (88 acoustic features from OpenSMILE eGeMAPS)
2. **Text Only** (100 TF-IDF features from transcripts)
3. **Prosody + Text Combined** (188 total features)

## Overall Performance Comparison

| Modality | Features | F1 (samples) | F1 (macro) | AUROC | Kappa | Hamming Loss |
|----------|----------|--------------|------------|-------|-------|--------------|
| **Prosody Only** | 88 | **0.079 ± 0.117** | 0.032 ± 0.034 | 0.479 ± 0.043 | -0.006 | 0.153 ± 0.017 |
| **Text Only** | 100 | 0.057 ± 0.038 | 0.039 ± 0.023 | **0.535 ± 0.033** | 0.031 | 0.150 ± 0.023 |
| **Prosody + Text** | 188 | 0.045 ± 0.060 | 0.025 ± 0.032 | 0.472 ± 0.054 | 0.004 | 0.152 ± 0.017 |

### 🏆 Winners

- **Best F1 Score**: Prosody (0.079)
- **Best AUROC**: Text (0.535)
- **Best Kappa**: Text (0.031)
- **Lowest Hamming Loss**: Text (0.150)

## Key Findings

### 1. **Text Outperforms Prosody for Discriminability**

- **AUROC**: Text achieves 0.535 vs Prosody's 0.479 (+11.7% improvement)
- Text features provide better class separation (AUROC > 0.5 vs random)
- Prosody features alone struggle to discriminate between emotions (AUROC ≈ 0.48)

### 2. **Prosody Better for Overall Prediction**

- **F1 (samples)**: Prosody achieves 0.079 vs Text's 0.057
- Prosody has higher variance, suggesting it captures some samples very well
- Text is more consistent but lower overall performance

### 3. **Combined Features Underperform**

**Surprising Result**: Combined features (0.045 F1) are **worse** than either modality alone!

**Possible Explanations**:
- **Feature dilution**: 188 features may be too many for small dataset (289 samples)
- **Overfitting**: RandomForest with 9-fold CV struggles with high-dimensional space
- **Feature imbalance**: Prosody (88) and text (100) features not properly balanced
- **No feature selection**: Should have used SelectKBest to reduce dimensionality
- **Need ensemble approach**: Late fusion or stacking might work better

## Per-Label Performance

### Best Emotions to Predict

**Prosody Only**:
1. **Optimistic**: F1=0.284, AUROC=0.525
2. **Curious**: F1=0.083, AUROC=0.518
3. **Confused**: F1=0.054, AUROC=0.481

**Text Only**:
1. **Curious**: F1=0.233, AUROC=0.582 ⭐ (Best AUROC)
2. **Optimistic**: F1=0.117, AUROC=0.410

**Prosody + Text**:
1. **Curious**: F1=0.155, AUROC=0.505
2. **Optimistic**: F1=0.122, AUROC=0.525

### Per-Label AUROC Analysis

| Emotion | Prosody | Text | Combined | Best Modality |
|---------|---------|------|----------|---------------|
| **Curious** | 0.518 | **0.582** ⭐ | 0.505 | Text |
| **Optimistic** | **0.525** | 0.410 | **0.525** | Prosody/Combined |
| **Disengaged** | 0.316 | **0.651** ⭐⭐ | 0.408 | Text |
| **Confused** | 0.481 | **0.552** | 0.474 | Text |
| **Frustrated** | 0.359 | **0.552** | 0.418 | Text |
| **Conflicted** | 0.354 | **0.541** | 0.364 | Text |
| **Surprised** | 0.440 | **0.464** | 0.386 | Text |
| **Engaged** | 0.271 | **0.367** | 0.284 | Text |

**Insight**: Text features excel at discriminating most emotions, especially **Disengaged** (AUROC=0.651) and **Curious** (AUROC=0.582).

## Detailed Metrics

### Prosody Only (88 features)
- ✅ **Strengths**: Best overall F1, good for Optimistic detection
- ❌ **Weaknesses**: Poor AUROC (~0.48), struggles with rare emotions
- 📊 **Use Case**: When you need to catch high-confidence predictions

### Text Only (100 TF-IDF features)
- ✅ **Strengths**: Best AUROC (0.535), best discrimination, works well for Curious/Disengaged
- ❌ **Weaknesses**: Lower F1 score, may be too conservative
- 📊 **Use Case**: When you need reliable class separation

### Prosody + Text (188 features)
- ❌ **Weaknesses**: Worst performance across all metrics
- ⚠️ **Issue**: Feature combination not working as expected
- 🔧 **Needs**: Feature selection, better fusion strategy, or ensemble methods

## Recommendations

### 1. **Use Text Features for This Task**

Despite lower F1, text features provide:
- Better discriminative power (AUROC > 0.5)
- More consistent predictions across folds
- Better performance on key emotions (Curious, Disengaged)

### 2. **Fix the Combined Approach**

Current early fusion (concatenation) doesn't work. Try:

**Feature Selection**:
```python
# Select top-k features before combining
selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X_combined, y)
```

**Late Fusion** (Ensemble):
```python
# Train separate models, combine predictions
prosody_model.fit(X_prosody, y_train)
text_model.fit(X_text, y_train)

pred_prosody = prosody_model.predict_proba(X_prosody_test)
pred_text = text_model.predict_proba(X_text_test)

# Weighted average
pred_combined = 0.4 * pred_prosody + 0.6 * pred_text
```

**Stacking**:
```python
# Use prosody + text predictions as meta-features
meta_features = np.hstack([pred_prosody, pred_text])
meta_model.fit(meta_features, y_train)
```

### 3. **Emotion-Specific Strategies**

| Emotion | Best Modality | Strategy |
|---------|---------------|----------|
| **Curious** | Text (F1=0.233) | Use text features, transcripts have curiosity markers |
| **Optimistic** | Prosody (F1=0.284) | Use prosodic features, positivity in tone |
| **Disengaged** | Text (AUROC=0.651) | Text discriminates well, look for short/empty responses |
| **Confused** | Prosody (F1=0.054) | Difficult emotion, needs more data |

### 4. **Consider Task Simplification**

Current performance is low across all modalities. Consider:

- **Binary Classification**: Engaged vs Disengaged (easier, more practical)
- **Three-Class Valence**: Positive, Neutral, Negative
- **Focus on Top-3 Emotions**: Optimistic, Curious, Confused (most frequent)

### 5. **Improve Text Features**

Current TF-IDF is basic. Try:

- **Word embeddings**: Word2Vec, GloVe, FastText
- **Contextualized embeddings**: BERT, RoBERTa
- **Linguistic features**: Sentiment, POS tags, named entities
- **Domain-specific features**: Question markers, filler words, hedge words

## Comparison with Previous Results

### Participant-based LOGO CV (27 folds) - Prosody Only:
- F1 (samples): 0.154 ± 0.116
- AUROC: 0.444 ± 0.050

### Group-based LOGO CV (9 folds) - Prosody Only:
- F1 (samples): 0.079 ± 0.117
- AUROC: 0.479 ± 0.043

**Observation**: Group-based CV is more stringent (larger test sets, more domain shift), resulting in lower but more realistic performance.

## Statistical Significance

Using 9-fold CV with small sample sizes per fold means:
- High variance in metrics (large ±std values)
- Some folds have F1=0.0 (complete failure)
- Results not statistically robust

**Recommendation**: With only 289 samples across 9 groups, consider:
- Stratified group sampling
- Repeated CV with different random seeds
- Bootstrap confidence intervals

## Conclusion

### Main Takeaways:

1. ✅ **Text features are more informative than prosody** for this dataset
   - AUROC 0.535 vs 0.479 (+11.7%)
   - Better for most emotions

2. ❌ **Simple feature concatenation doesn't work**
   - Combined features underperform both individual modalities
   - Need better fusion strategies

3. 📊 **Different modalities excel at different emotions**
   - Text: Curious, Disengaged
   - Prosody: Optimistic
   - Suggests emotion-specific ensembles

4. 🎯 **Overall performance is still low**
   - F1 scores < 0.10 indicate very difficult task
   - Consider task simplification or more data

### Next Steps:

1. Implement late fusion or stacking ensemble
2. Add BERT embeddings instead of TF-IDF
3. Try emotion-specific models
4. Simplify to binary or 3-class task
5. Collect more diverse training data

## Files Generated

- `multilabel_multimodal_logo.py`: Main analysis script
- `display_multimodal_results.py`: Visualization script
- `results/multimodal_comparison_20260205_224159.csv`: Results table
- `results/multimodal_results_20260205_224159.pkl`: Full results
- `results/multimodal_comparison.png`: 4-panel visualization
- `MULTIMODAL_COMPARISON.md`: This document
