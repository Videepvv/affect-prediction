"""
Compare affective state prediction across different window sizes
Compares 5s, 10s, and 20s time windows for multi-label classification
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load results
df_5s = pd.read_csv('results/window_5s_comparison_20260205_230227.csv')
df_10s = pd.read_csv('results/window_10s_comparison_20260205_225823.csv')

# Load 20s results (from multimodal comparison) - note: different format
df_20s_raw = pd.read_csv('results/multimodal_comparison_20260205_224713.csv')

# Parse the 20s data which has format: "value ± std"
def parse_value_std(s):
    """Parse 'mean ± std' string into mean and std"""
    parts = str(s).split('±')
    mean = float(parts[0].strip())
    std = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() != 'nan' else 0.0
    return mean, std

df_20s_clean_data = []
modality_map = {'Prosody Only': 'prosody', 'Text Only': 'text', 'Prosody + Text': 'combined'}

for _, row in df_20s_raw.iterrows():
    modality = modality_map.get(row['Modality'], row['Modality'])
    f1_mean, f1_std = parse_value_std(row['F1 (samples)'])
    auroc_mean, auroc_std = parse_value_std(row['AUROC'])
    hamming_mean, hamming_std = parse_value_std(row['Hamming Loss'])
    
    df_20s_clean_data.append({
        'WindowSize': '20s',
        'Modality': modality,
        'F1_samples_mean': f1_mean,
        'F1_samples_std': f1_std,
        'AUROC_macro_mean': auroc_mean,
        'AUROC_macro_std': auroc_std,
        'Hamming_loss_mean': hamming_mean,
        'Hamming_loss_std': hamming_std
    })

df_20s_clean = pd.DataFrame(df_20s_clean_data)

# Define emotion labels
emotion_labels = ['Optimistic', 'Curious', 'Confused', 'Conflicted', 
                  'Surprised', 'Disengaged', 'Frustrated', 'Engaged']

# Combine all data
df_all = pd.concat([df_5s, df_10s, df_20s_clean], ignore_index=True)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Affective State Prediction: Window Size Comparison\n(5s vs 10s vs 20s with BERT embeddings)', 
             fontsize=16, fontweight='bold')

# Define colors
colors = {'prosody': '#2ecc71', 'text': '#3498db', 'combined': '#e74c3c'}

# Plot 1: F1 Score by Window Size
ax = axes[0, 0]
window_order = ['5s', '10s', '20s']
modality_order = ['prosody', 'text', 'combined']

x = np.arange(len(window_order))
width = 0.25

for i, modality in enumerate(modality_order):
    data = df_all[df_all['Modality'] == modality]
    means = [data[data['WindowSize'] == w]['F1_samples_mean'].values[0] for w in window_order]
    stds = [data[data['WindowSize'] == w]['F1_samples_std'].values[0] for w in window_order]
    ax.bar(x + i*width, means, width, yerr=stds, label=modality.capitalize(), 
           color=colors[modality], alpha=0.8, capsize=5)

ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score (samples)', fontsize=12, fontweight='bold')
ax.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(window_order)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: AUROC by Window Size
ax = axes[0, 1]
for i, modality in enumerate(modality_order):
    data = df_all[df_all['Modality'] == modality]
    means = [data[data['WindowSize'] == w]['AUROC_macro_mean'].values[0] for w in window_order]
    stds = [data[data['WindowSize'] == w]['AUROC_macro_std'].values[0] for w in window_order]
    ax.bar(x + i*width, means, width, yerr=stds, label=modality.capitalize(), 
           color=colors[modality], alpha=0.8, capsize=5)

ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('AUROC (macro)', fontsize=12, fontweight='bold')
ax.set_title('AUROC Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(window_order)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Hamming Loss by Window Size
ax = axes[0, 2]
for i, modality in enumerate(modality_order):
    data = df_all[df_all['Modality'] == modality]
    means = [data[data['WindowSize'] == w]['Hamming_loss_mean'].values[0] for w in window_order]
    stds = [data[data['WindowSize'] == w]['Hamming_loss_std'].values[0] for w in window_order]
    ax.bar(x + i*width, means, width, yerr=stds, label=modality.capitalize(), 
           color=colors[modality], alpha=0.8, capsize=5)

ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Hamming Loss', fontsize=12, fontweight='bold')
ax.set_title('Hamming Loss (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(window_order)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4-6: Per-emotion AUROC heatmaps for each window size
for idx, window in enumerate(window_order):
    ax = axes[1, idx]
    
    # Extract per-label AUROC for this window
    auroc_data = []
    for modality in modality_order:
        row_data = []
        for label in emotion_labels:
            col_name = f'AUROC_{label}_mean'
            if col_name in df_all.columns:
                value = df_all[(df_all['WindowSize'] == window) & 
                              (df_all['Modality'] == modality)][col_name].values
                row_data.append(value[0] if len(value) > 0 else np.nan)
            else:
                row_data.append(np.nan)
        auroc_data.append(row_data)
    
    auroc_df = pd.DataFrame(auroc_data, 
                           columns=[l.replace(' ', '\n') for l in emotion_labels],
                           index=[m.capitalize() for m in modality_order])
    
    sns.heatmap(auroc_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                vmin=0.3, vmax=0.7, ax=ax, cbar_kws={'label': 'AUROC'})
    ax.set_title(f'Per-Emotion AUROC ({window} windows)', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Modality', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('window_size_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: window_size_comparison.png")

# Create summary table
print("\n" + "="*80)
print("WINDOW SIZE COMPARISON SUMMARY")
print("="*80)

summary_data = []
for window in window_order:
    for modality in modality_order:
        row = df_all[(df_all['WindowSize'] == window) & (df_all['Modality'] == modality)].iloc[0]
        summary_data.append({
            'Window': window,
            'Modality': modality.capitalize(),
            'F1': f"{row['F1_samples_mean']:.3f}±{row['F1_samples_std']:.3f}",
            'AUROC': f"{row['AUROC_macro_mean']:.3f}±{row['AUROC_macro_std']:.3f}",
            'Hamming': f"{row['Hamming_loss_mean']:.3f}±{row['Hamming_loss_std']:.3f}"
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Best performance by metric
print("\n" + "="*80)
print("BEST PERFORMANCE BY METRIC")
print("="*80)

# Best F1
best_f1_idx = df_all['F1_samples_mean'].idxmax()
best_f1 = df_all.iloc[best_f1_idx]
print(f"\nBest F1 Score:")
print(f"  {best_f1['WindowSize']} window, {best_f1['Modality'].capitalize()} modality")
print(f"  F1 = {best_f1['F1_samples_mean']:.3f}±{best_f1['F1_samples_std']:.3f}")

# Best AUROC
best_auroc_idx = df_all['AUROC_macro_mean'].idxmax()
best_auroc = df_all.iloc[best_auroc_idx]
print(f"\nBest AUROC:")
print(f"  {best_auroc['WindowSize']} window, {best_auroc['Modality'].capitalize()} modality")
print(f"  AUROC = {best_auroc['AUROC_macro_mean']:.3f}±{best_auroc['AUROC_macro_std']:.3f}")

# Best Hamming Loss (lowest)
best_hamming_idx = df_all['Hamming_loss_mean'].idxmin()
best_hamming = df_all.iloc[best_hamming_idx]
print(f"\nLowest Hamming Loss:")
print(f"  {best_hamming['WindowSize']} window, {best_hamming['Modality'].capitalize()} modality")
print(f"  Hamming = {best_hamming['Hamming_loss_mean']:.3f}±{best_hamming['Hamming_loss_std']:.3f}")

# Trend analysis
print("\n" + "="*80)
print("TREND ANALYSIS")
print("="*80)

for modality in modality_order:
    print(f"\n{modality.upper()} Modality:")
    data = df_all[df_all['Modality'] == modality].sort_values('WindowSize')
    
    f1_values = data['F1_samples_mean'].values
    auroc_values = data['AUROC_macro_mean'].values
    
    f1_trend = "increasing" if f1_values[-1] > f1_values[0] else "decreasing"
    auroc_trend = "increasing" if auroc_values[-1] > auroc_values[0] else "decreasing"
    
    print(f"  F1: {f1_values[0]:.3f} (5s) → {f1_values[1]:.3f} (10s) → {f1_values[2]:.3f} (20s) [{f1_trend}]")
    print(f"  AUROC: {auroc_values[0]:.3f} (5s) → {auroc_values[1]:.3f} (10s) → {auroc_values[2]:.3f} (20s) [{auroc_trend}]")

# Save comprehensive data
df_all.to_csv('results/window_size_comprehensive_comparison.csv', index=False)
print("\n" + "="*80)
print("Saved comprehensive comparison to: results/window_size_comprehensive_comparison.csv")
print("="*80)

plt.show()
