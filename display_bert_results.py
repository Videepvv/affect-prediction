"""Display and visualize BERT-based multimodal comparison results"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_file = '/home/videep/research/predicting-affectivestates/results/multimodal_results_20260205_224713.pkl'
results = joblib.load(results_file)

print("="*80)
print("MULTI-MODAL COMPARISON: PROSODY vs BERT vs COMBINED")
print("="*80)

# Display comparison table
comparison_df = pd.DataFrame(results['comparison'])
print("\n" + comparison_df.to_string(index=False))

# Extract mean and std for each modality
print("\n" + "="*80)
print("DETAILED PERFORMANCE METRICS")
print("="*80)

modalities = ['prosody', 'text', 'combined']
mod_names = ['Prosody Only', 'BERT Only', 'Prosody + BERT']

for mod, name in zip(modalities, mod_names):
    print(f"\n{name}:")
    print("-" * 60)
    fold_df = pd.DataFrame(results[mod]['fold_results'])
    
    print(f"  F1 (samples):    {fold_df['f1_samples'].mean():.4f} ± {fold_df['f1_samples'].std():.4f}")
    print(f"  F1 (macro):      {fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}")
    print(f"  AUROC (macro):   {fold_df['auroc_macro'].mean():.4f} ± {fold_df['auroc_macro'].std():.4f}")
    print(f"  Hamming Loss:    {fold_df['hamming_loss'].mean():.4f} ± {fold_df['hamming_loss'].std():.4f}")
    print(f"  Subset Accuracy: {fold_df['subset_accuracy'].mean():.4f} ± {fold_df['subset_accuracy'].std():.4f}")

# Per-label comparison
print("\n" + "="*80)
print("PER-LABEL METRICS COMPARISON")
print("="*80)

labels = results['label_classes']

print(f"\n{'Label':<15} {'Modality':<18} {'F1':<10} {'AUROC':<10} {'Kappa':<10}")
print("-" * 63)

for label in labels:
    for mod, name in zip(modalities, mod_names):
        overall = results[mod]['overall']
        f1 = overall['per_class_f1'][label]
        auroc = overall['per_class_auroc'][label] if 'per_class_auroc' in overall else 0.0
        kappa = overall['per_class_kappa'][label]
        
        mod_short = name.split()[0] if len(name.split()) == 2 else name[:12]
        print(f"{label:<15} {mod_short:<18} {f1:<10.4f} {auroc:<10.4f} {kappa:<10.4f}")
    print()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. F1 Scores Comparison
ax = axes[0, 0]
f1_data = []
for mod, name in zip(modalities, mod_names):
    fold_df = pd.DataFrame(results[mod]['fold_results'])
    f1_data.extend([(name, val) for val in fold_df['f1_samples']])

f1_df = pd.DataFrame(f1_data, columns=['Modality', 'F1 (samples)'])
sns.boxplot(data=f1_df, x='Modality', y='F1 (samples)', ax=ax)
ax.set_title('F1 Score (samples) by Modality', fontsize=14, fontweight='bold')
ax.set_xlabel('Modality', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# 2. AUROC Comparison
ax = axes[0, 1]
auroc_data = []
for mod, name in zip(modalities, mod_names):
    fold_df = pd.DataFrame(results[mod]['fold_results'])
    auroc_data.extend([(name, val) for val in fold_df['auroc_macro']])

auroc_df = pd.DataFrame(auroc_data, columns=['Modality', 'AUROC (macro)'])
sns.boxplot(data=auroc_df, x='Modality', y='AUROC (macro)', ax=ax)
ax.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
ax.set_title('AUROC (macro) by Modality - BERT Embeddings', fontsize=14, fontweight='bold')
ax.set_xlabel('Modality', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3. Per-Label F1 Comparison (Heatmap)
ax = axes[1, 0]
f1_matrix = []
for label in labels:
    row = []
    for mod in modalities:
        overall = results[mod]['overall']
        f1 = overall['per_class_f1'][label]
        row.append(f1)
    f1_matrix.append(row)

f1_matrix = np.array(f1_matrix)
sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
           xticklabels=mod_names, yticklabels=labels, ax=ax, cbar_kws={'label': 'F1 Score'})
ax.set_title('Per-Label F1 Score Heatmap (BERT)', fontsize=14, fontweight='bold')
ax.set_xlabel('Modality', fontsize=12)
ax.set_ylabel('Emotion Label', fontsize=12)

# 4. Per-Label AUROC Comparison (Heatmap)
ax = axes[1, 1]
auroc_matrix = []
for label in labels:
    row = []
    for mod in modalities:
        overall = results[mod]['overall']
        auroc = overall['per_class_auroc'][label] if 'per_class_auroc' in overall else 0.5
        row.append(auroc)
    auroc_matrix.append(row)

auroc_matrix = np.array(auroc_matrix)
sns.heatmap(auroc_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
           xticklabels=mod_names, yticklabels=labels, ax=ax, cbar_kws={'label': 'AUROC'})
ax.set_title('Per-Label AUROC Heatmap (BERT)', fontsize=14, fontweight='bold')
ax.set_xlabel('Modality', fontsize=12)
ax.set_ylabel('Emotion Label', fontsize=12)

plt.tight_layout()
plt.savefig('/home/videep/research/predicting-affectivestates/results/multimodal_bert_comparison.png', 
           dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("VISUALIZATION SAVED")
print("="*80)
print("📊 results/multimodal_bert_comparison.png")
plt.close()

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Extract mean metrics
prosody_f1 = pd.DataFrame(results['prosody']['fold_results'])['f1_samples'].mean()
bert_f1 = pd.DataFrame(results['text']['fold_results'])['f1_samples'].mean()
combined_f1 = pd.DataFrame(results['combined']['fold_results'])['f1_samples'].mean()

prosody_auroc = pd.DataFrame(results['prosody']['fold_results'])['auroc_macro'].mean()
bert_auroc = pd.DataFrame(results['text']['fold_results'])['auroc_macro'].mean()
combined_auroc = pd.DataFrame(results['combined']['fold_results'])['auroc_macro'].mean()

print(f"\nBest F1 (samples):")
best_f1 = max(prosody_f1, bert_f1, combined_f1)
best_f1_mod = ['Prosody', 'BERT', 'Combined'][np.argmax([prosody_f1, bert_f1, combined_f1])]
print(f"  Winner: {best_f1_mod} with {best_f1:.4f}")

print(f"\nBest AUROC (macro):")
best_auroc = max(prosody_auroc, bert_auroc, combined_auroc)
best_auroc_mod = ['Prosody', 'BERT', 'Combined'][np.argmax([prosody_auroc, bert_auroc, combined_auroc])]
print(f"  Winner: {best_auroc_mod} with {best_auroc:.4f}")

print(f"\nProsody vs BERT:")
if prosody_f1 > 0:
    print(f"  BERT F1 change: {((bert_f1 - prosody_f1) / prosody_f1 * 100):.1f}%")
print(f"  BERT improves AUROC by {((bert_auroc - prosody_auroc) / prosody_auroc * 100):.1f}%")

print(f"\nCombined vs Individual:")
print(f"  Combined F1: {combined_f1:.4f}")
print(f"  Best individual F1: {max(prosody_f1, bert_f1):.4f}")
print(f"  Combined AUROC: {combined_auroc:.4f}")
print(f"  Best individual AUROC: {max(prosody_auroc, bert_auroc):.4f}")

if combined_auroc > max(prosody_auroc, bert_auroc):
    print(f"  ✓ Combined AUROC is BETTER by {((combined_auroc - max(prosody_auroc, bert_auroc)) / max(prosody_auroc, bert_auroc) * 100):.1f}%")
else:
    print(f"  ✗ Combined AUROC is WORSE by {((max(prosody_auroc, bert_auroc) - combined_auroc) / max(prosody_auroc, bert_auroc) * 100):.1f}%")

print(f"\nBest emotions to predict:")
for mod, name in zip(modalities, mod_names):
    overall = results[mod]['overall']
    best_labels = sorted(overall['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:3]
    best_str = ', '.join([f'{l}({f:.3f})' for l, f in best_labels if f > 0])
    if best_str:
        print(f"  {name}: {best_str}")
    else:
        print(f"  {name}: No labels with F1 > 0")

print("\n" + "="*80)
print("COMPARISON: TF-IDF vs BERT EMBEDDINGS")
print("="*80)
print("\nFeature Dimensions:")
print(f"  TF-IDF features: 100 (sparse bag-of-words)")
print(f"  BERT features: 768 (dense contextual embeddings)")
print(f"\nPerformance:")
print(f"  TF-IDF AUROC: 0.535 (from previous run)")
print(f"  BERT AUROC: {bert_auroc:.3f}")
if bert_auroc > 0.535:
    print(f"  ✓ BERT is {((bert_auroc - 0.535) / 0.535 * 100):.1f}% better!")
else:
    print(f"  ✗ BERT is {((0.535 - bert_auroc) / 0.535 * 100):.1f}% worse")

print("\n" + "="*80)
