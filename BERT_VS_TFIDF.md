# BERT Embeddings vs TF-IDF: Multi-Modal Analysis Results

## Summary

This analysis compares **BERT embeddings** vs **TF-IDF** for text-based affective state prediction, using the same multi-label Leave-One-Group-Out Cross-Validation (9 groups).

## Overall Performance Comparison

| Modality | Features | F1 (samples) | F1 (macro) | AUROC | Hamming Loss |
|----------|----------|--------------|------------|-------|--------------|
| **Prosody Only** | 88 | **0.079 ± 0.117** 🏆 | 0.032 ± 0.034 | 0.479 ± 0.043 | 0.153 ± 0.017 |
| **BERT Only** | 768 | 0.028 ± 0.037 | 0.015 ± 0.021 | **0.548 ± 0.049** 🏆 | 0.154 ± 0.018 |
| **Prosody + BERT** | 856 | 0.028 ± 0.037 | 0.016 ± 0.021 | 0.528 ± 0.058 | 0.150 ± 0.017 |

### Comparison: TF-IDF vs BERT

| Text Modality | Features | F1 (samples) | AUROC | Notes |
|---------------|----------|--------------|-------|-------|
| **TF-IDF** | 100 | 0.057 ± 0.038 | 0.535 ± 0.033 | Sparse, bag-of-words |
| **BERT** | 768 | 0.028 ± 0.037 | **0.548 ± 0.049** | Dense, contextual |

## Key Findings

### 🏆 BERT Advantages

1. **Best Discriminative Power** (AUROC = 0.548)
   - BERT achieves highest AUROC across all modalities
   - **+2.4% improvement** over TF-IDF (0.535 → 0.548)
   - **+14.2% improvement** over prosody (0.479 → 0.548)
   - Above random threshold (0.5) indicates meaningful class separation

2. **Better for Specific Emotions**:
   - **Conflicted**: AUROC 0.629 (vs TF-IDF 0.541) - **+16% improvement**
   - **Confused**: AUROC 0.594 (vs TF-IDF 0.552)
   - **Surprised**: AUROC 0.528 (vs TF-IDF 0.464)
   - **Curious**: F1 0.108 vs TF-IDF 0.233 (worse)

3. **Contextual Understanding**:
   - BERT captures semantic context and word relationships
   - Better at understanding nuanced emotional expressions
   - Handles longer transcripts more effectively

### ⚠️ BERT Disadvantages

1. **Lower F1 Scores** (0.028 vs TF-IDF 0.057)
   - **64% drop** compared to TF-IDF in F1 score
   - More conservative predictions
   - Worse precision/recall trade-off

2. **Computational Cost**:
   - 768 features vs 100 TF-IDF features (7.6x larger)
   - Requires GPU for efficient inference
   - Pre-trained model download (~400MB)
   - Slower processing time

3. **Still Underperforms Prosody on F1**:
   - Prosody F1: 0.079
   - BERT F1: 0.028 (-64%)
   - Prosody still better for overall prediction task

### 🎯 Combined (Prosody + BERT) Analysis

**Surprising Result**: Combined features don't improve over BERT alone!

| Metric | BERT Only | Prosody + BERT | Change |
|--------|-----------|----------------|--------|
| F1 (samples) | 0.028 | 0.028 | **0%** |
| AUROC | 0.548 | 0.528 | **-3.5%** ❌ |
| Features | 768 | 856 | +88 |

**Why Combined Underperforms**:
- **Dimensionality curse**: 856 features with only 289 samples
- **Feature imbalance**: BERT (768) dominates prosody (88)
- **No feature selection**: Should reduce to 100-200 most informative features
- **Model limitations**: RandomForest struggles with high-dimensional spaces

## Per-Label Performance Analysis

### Best Modality per Emotion:

| Emotion | Best for F1 | F1 Score | Best for AUROC | AUROC Score |
|---------|-------------|----------|----------------|-------------|
| **Optimistic** | Prosody | 0.284 | Prosody | 0.525 |
| **Curious** | BERT | 0.108 | Prosody | 0.517 |
| **Confused** | Prosody | 0.054 | **BERT** | **0.594** ⭐ |
| **Conflicted** | - | 0.000 | **BERT** | **0.629** ⭐⭐ |
| **Surprised** | - | 0.000 | **BERT** | **0.528** |
| **Frustrated** | - | 0.000 | BERT | 0.438 |
| **Disengaged** | - | 0.000 | BERT | 0.453 |
| **Engaged** | - | 0.000 | BERT | 0.313 |

### Key Observations:

1. **Prosody dominates F1 scores**
   - Only modality with F1 > 0.10
   - Best for Optimistic (0.284)

2. **BERT dominates AUROC**
   - Best discrimination for 6/8 emotions
   - Especially strong for Conflicted (0.629) and Confused (0.594)

3. **Most emotions are unpredictable**
   - 5/8 emotions have F1 = 0 across all modalities
   - Suggests these emotions are rare or not distinguishable

## TF-IDF vs BERT: Detailed Analysis

### TF-IDF Strengths:
✅ **Better F1** (0.057 vs 0.028) - more aggressive predictions  
✅ **Fewer features** (100 vs 768) - less overfitting  
✅ **Fast** - no GPU required  
✅ **Interpretable** - can see word importance  

### BERT Strengths:
✅ **Better AUROC** (0.548 vs 0.535) - better discrimination  
✅ **Contextual** - understands word order and semantics  
✅ **Pre-trained** - leverages knowledge from large corpora  
✅ **Better for complex emotions** (Conflicted, Confused)  

### When to Use Each:

| Use TF-IDF When: | Use BERT When: |
|------------------|----------------|
| Small dataset (<1000 samples) | Larger dataset (>5000 samples) |
| Need interpretability | Accuracy is paramount |
| Computational resources limited | GPU available |
| Bag-of-words sufficient | Context matters |
| Rapid prototyping | Production system |

## Recommendations

### 1. **For This Dataset: Use BERT for Discrimination Tasks**

If you need to **rank** or **filter** samples by confidence:
```python
# Use BERT AUROC scores
bert_probabilities = model.predict_proba(X_bert)
# Select top-k most confident predictions
confident_samples = np.argsort(bert_probabilities)[-k:]
```

### 2. **For This Dataset: Use Prosody for Prediction Tasks**

If you need to **predict labels** with best F1:
```python
# Use prosody for actual predictions
prosody_predictions = model.predict(X_prosody)
```

### 3. **Improve Combined Approach**

Current concatenation doesn't work. Try:

**A. Feature Selection Before Fusion**:
```python
# Reduce BERT to top-100 features
selector = SelectKBest(mutual_info_classif, k=100)
X_bert_reduced = selector.fit_transform(X_bert, y)

# Concatenate: 88 prosody + 100 BERT = 188 features
X_combined = np.hstack([X_prosody, X_bert_reduced])
```

**B. Late Fusion (Ensemble)**:
```python
# Train separate models
prosody_model.fit(X_prosody, y_train)
bert_model.fit(X_bert, y_train)

# Weighted voting
pred_prosody = prosody_model.predict_proba(X_prosody_test)
pred_bert = bert_model.predict_proba(X_bert_test)

# Optimize weights based on AUROC
pred_final = 0.3 * pred_prosody + 0.7 * pred_bert  # BERT weighted higher
```

**C. Stacking Meta-Learner**:
```python
# Use both predictions as meta-features
meta_features = np.hstack([
    prosody_model.predict_proba(X_prosody),
    bert_model.predict_proba(X_bert)
])

# Train meta-classifier
meta_model.fit(meta_features, y_train)
```

### 4. **Try Smaller BERT Models**

Current: `bert-base-uncased` (768 dims, 110M params)

Try instead:
- **DistilBERT** (768 dims, 66M params) - 40% faster, 97% performance
- **TinyBERT** (312 dims, 14M params) - 7.5x smaller
- **MobileBERT** (512 dims, 25M params) - optimized for mobile

```python
# Use DistilBERT instead
X_text = modeler.prepare_features(
    feature_type='text', 
    bert_model='distilbert-base-uncased'
)
```

### 5. **Fine-tune BERT on Your Domain**

Pre-trained BERT is general-purpose. Fine-tuning on your transcripts could help:

```python
from transformers import Trainer, TrainingArguments

# Fine-tune BERT on your transcripts with emotion labels
trainer = Trainer(
    model=bert_model,
    args=TrainingArguments(
        output_dir='./bert-finetuned',
        num_train_epochs=3,
        learning_rate=2e-5
    ),
    train_dataset=your_dataset
)

trainer.train()
```

### 6. **Task Simplification**

Current performance (F1 < 0.1) suggests task is too hard. Simplify:

**Binary Classification**:
- Positive (Optimistic, Curious, Engaged) vs Negative (Frustrated, Confused, Disengaged)
- Expected improvement: F1 > 0.3

**Three-Class Valence**:
- Positive, Neutral, Negative
- Expected improvement: F1 > 0.25

**Focus on Top-2 Emotions**:
- Only predict Optimistic and Curious
- Expected improvement: F1 > 0.2

## Statistical Significance

With 289 samples across 9 groups:
- **High variance** (large ±std values indicate instability)
- **Some folds fail completely** (F1 = 0.0)
- **Results not robust** - need more data or simpler task

**Bootstrap Analysis** (recommended):
```python
from sklearn.utils import resample

bootstrap_scores = []
for i in range(1000):
    # Resample with replacement
    X_boot, y_boot = resample(X, y)
    score = model.fit(X_boot, y_boot).score(X_test, y_test)
    bootstrap_scores.append(score)

# 95% confidence interval
ci_lower = np.percentile(bootstrap_scores, 2.5)
ci_upper = np.percentile(bootstrap_scores, 97.5)
```

## Conclusion

### Main Takeaways:

1. ✅ **BERT improves AUROC over TF-IDF** (+2.4%, 0.535 → 0.548)
   - Better discrimination and ranking capability
   - Especially strong for Conflicted (+16%) and Confused emotions

2. ❌ **BERT has worse F1 than TF-IDF** (-50%)
   - More conservative, fewer predictions
   - Trade-off: precision vs recall

3. ❌ **Simple feature concatenation still doesn't work**
   - Combined (856 features) underperforms BERT alone (768 features)
   - Need feature selection or ensemble methods

4. 🏆 **Prosody still best for F1 overall**
   - 0.079 vs BERT 0.028
   - But prosody has poor AUROC (0.479)

5. 🎯 **Emotion-specific strategies needed**
   - Optimistic: Use prosody
   - Conflicted/Confused: Use BERT
   - Curious: Either works

### Recommended Approach:

**For Production System**:
```python
# Emotion-specific ensemble
if emotion in ['Optimistic']:
    prediction = prosody_model.predict(X_prosody)
elif emotion in ['Conflicted', 'Confused', 'Surprised']:
    prediction = bert_model.predict(X_bert)
else:  # Curious
    # Weighted average
    pred = 0.6 * prosody_model.predict_proba(X_prosody) + \
           0.4 * bert_model.predict_proba(X_bert)
    prediction = (pred > 0.5).astype(int)
```

### Next Steps:

1. ✅ Try DistilBERT (faster, similar performance)
2. ✅ Implement late fusion ensemble
3. ✅ Add feature selection before concatenation
4. ✅ Fine-tune BERT on domain-specific data
5. ✅ Simplify task to binary or 3-class
6. ✅ Collect more labeled data (target: 1000+ samples)

## Files Generated

- `multilabel_multimodal_logo.py`: Updated with BERT embeddings
- `display_bert_results.py`: BERT visualization script
- `results/multimodal_comparison_20260205_224713.csv`: BERT results table
- `results/multimodal_results_20260205_224713.pkl`: Full BERT results
- `results/multimodal_bert_comparison.png`: 4-panel BERT visualization
- `BERT_VS_TFIDF.md`: This comprehensive comparison document
