import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def print_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


print_with_timestamp("Starting Polynomial Regression Model Comparison")

# Check which result files exist
result_files = {
    'Linear': 'linear_training_results.csv',
    'Ridge': 'ridge_training_results.csv',
    'Lasso': 'lasso_training_results.csv',
    'ElasticNet': 'elasticnet_training_results.csv'
}

available_models = {}
for model_name, filename in result_files.items():
    if os.path.exists(filename):
        available_models[model_name] = pd.read_csv(filename)
        print_with_timestamp(f"✅ Found results for {model_name}: {filename}")
    else:
        print_with_timestamp(f"❌ Missing results for {model_name}: {filename}")

if not available_models:
    print_with_timestamp(
        "❌ No result files found! Please run the individual model scripts first.")
    exit(1)

print_with_timestamp(
    f"📊 Comparing {len(available_models)} models: {list(available_models.keys())}")

# Calculate summary statistics for each model
model_summaries = []

for model_name, results_df in available_models.items():
    summary = {
        'Model': model_name,
        'Avg_Val_MAPE': results_df['val_mape'].mean(),
        'Std_Val_MAPE': results_df['val_mape'].std(),
        'Best_Val_MAPE': results_df['val_mape'].min(),
        'Avg_Val_R2': results_df['val_r2'].mean(),
        'Best_Val_R2': results_df['val_r2'].max(),
        'Avg_Training_Time': results_df['training_time'].mean(),
        'Total_Training_Time': results_df['training_time'].sum(),
        'Best_Fold': results_df.loc[results_df['val_mape'].idxmin(), 'fold']
    }

    # Add model-specific metrics
    if 'alpha' in results_df.columns:
        best_row = results_df.loc[results_df['val_mape'].idxmin()]
        summary['Best_Alpha'] = best_row['alpha']

    if 'l1_ratio' in results_df.columns:
        best_row = results_df.loc[results_df['val_mape'].idxmin()]
        summary['Best_L1_Ratio'] = best_row['l1_ratio']

    if 'sparsity' in results_df.columns:
        summary['Avg_Sparsity'] = results_df['sparsity'].mean()
        summary['Features_Kept_Pct'] = 100 * \
            (1 - results_df['sparsity'].mean())

    model_summaries.append(summary)

# Convert to DataFrame and sort by performance
summary_df = pd.DataFrame(model_summaries)
summary_df = summary_df.sort_values('Best_Val_MAPE')

print_with_timestamp("\n📊 MODEL COMPARISON SUMMARY")
print_with_timestamp("=" * 80)

# Print performance ranking
print_with_timestamp("🏆 Performance Ranking (by Best Validation MAPE):")
for i, row in summary_df.iterrows():
    rank = summary_df.index.get_loc(i) + 1
    print_with_timestamp(
        f"  #{rank}: {row['Model']:<12} {row['Best_Val_MAPE']:.2f}% MAPE (Avg: {row['Avg_Val_MAPE']:.2f}% ± {row['Std_Val_MAPE']:.2f}%)")

print_with_timestamp(f"\n⚡ Training Time Comparison:")
for i, row in summary_df.iterrows():
    print_with_timestamp(
        f"  {row['Model']:<12} {row['Total_Training_Time']:.1f}s total ({row['Avg_Training_Time']:.1f}s per fold)")

# Create comprehensive comparison visualization
print_with_timestamp("\n📊 Creating comparison visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Polynomial Regression Model Comparison',
             fontsize=16, fontweight='bold')

# Plot 1: Performance comparison (MAPE)
ax = axes[0, 0]
models = summary_df['Model'].tolist()
best_mapes = summary_df['Best_Val_MAPE'].tolist()
avg_mapes = summary_df['Avg_Val_MAPE'].tolist()
std_mapes = summary_df['Std_Val_MAPE'].tolist()

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, best_mapes, alpha=0.7, color=[
              'blue', 'green', 'orange', 'red'][:len(models)])
ax.errorbar(x_pos, avg_mapes, yerr=std_mapes, fmt='o',
            color='black', capsize=5, label='Avg ± Std')

ax.set_xlabel('Model')
ax.set_ylabel('Validation MAPE (%)')
ax.set_title('Best Validation MAPE by Model')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, best_mapes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom')

# Plot 2: R² comparison
ax = axes[0, 1]
best_r2s = summary_df['Best_Val_R2'].tolist()
avg_r2s = summary_df['Avg_Val_R2'].tolist()

bars = ax.bar(x_pos, best_r2s, alpha=0.7, color=[
              'blue', 'green', 'orange', 'red'][:len(models)])
ax.scatter(x_pos, avg_r2s, color='black', s=50, zorder=5, label='Average R²')

ax.set_xlabel('Model')
ax.set_ylabel('Validation R²')
ax.set_title('Best Validation R² by Model')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars, best_r2s):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom')

# Plot 3: Training time comparison
ax = axes[0, 2]
training_times = summary_df['Total_Training_Time'].tolist()

bars = ax.bar(x_pos, training_times, alpha=0.7, color=[
              'blue', 'green', 'orange', 'red'][:len(models)])

ax.set_xlabel('Model')
ax.set_ylabel('Total Training Time (seconds)')
ax.set_title('Training Time Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45)
ax.grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars, training_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}s', ha='center', va='bottom')

# Plot 4: Performance vs Training Time
ax = axes[1, 0]
colors = ['blue', 'green', 'orange', 'red']
for i, (_, row) in enumerate(summary_df.iterrows()):
    ax.scatter(row['Total_Training_Time'], row['Best_Val_MAPE'],
               s=100, alpha=0.7, color=colors[i], label=row['Model'])

ax.set_xlabel('Total Training Time (seconds)')
ax.set_ylabel('Best Validation MAPE (%)')
ax.set_title('Performance vs Training Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Performance by fold for all models
ax = axes[1, 1]
for model_name, results_df in available_models.items():
    folds = results_df['fold'].tolist()
    val_mapes = results_df['val_mape'].tolist()
    ax.plot(folds, val_mapes, 'o-', label=model_name,
            linewidth=2, markersize=6)

ax.set_xlabel('Fold')
ax.set_ylabel('Validation MAPE (%)')
ax.set_title('Cross-Validation Performance')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Model characteristics
ax = axes[1, 2]
ax.axis('off')

# Create summary text
summary_text = "Model Comparison Summary\n"
summary_text += "=" * 30 + "\n\n"

best_model = summary_df.iloc[0]
summary_text += f"🏆 BEST MODEL: {best_model['Model']}\n"
summary_text += f"• Best MAPE: {best_model['Best_Val_MAPE']:.2f}%\n"
summary_text += f"• Best R²: {best_model['Best_Val_R2']:.4f}\n"
summary_text += f"• Training Time: {best_model['Total_Training_Time']:.1f}s\n\n"

summary_text += "📊 ALL MODELS:\n"
for _, row in summary_df.iterrows():
    summary_text += f"• {row['Model']:<12}: {row['Best_Val_MAPE']:.2f}% MAPE\n"

summary_text += f"\n⚡ FASTEST: {summary_df.loc[summary_df['Total_Training_Time'].idxmin(), 'Model']}\n"
summary_text += f"⚡ SLOWEST: {summary_df.loc[summary_df['Total_Training_Time'].idxmax(), 'Model']}\n"

# Add regularization info if available
if 'Best_Alpha' in summary_df.columns:
    summary_text += f"\n🎯 REGULARIZATION:\n"
    for _, row in summary_df.iterrows():
        if pd.notna(row.get('Best_Alpha')):
            alpha_str = f"α={row['Best_Alpha']}"
            if pd.notna(row.get('Best_L1_Ratio')):
                alpha_str += f", L1={row['Best_L1_Ratio']:.1f}"
            summary_text += f"• {row['Model']:<12}: {alpha_str}\n"

# Add sparsity info if available
if 'Features_Kept_Pct' in summary_df.columns:
    summary_text += f"\n🔍 FEATURE SELECTION:\n"
    for _, row in summary_df.iterrows():
        if pd.notna(row.get('Features_Kept_Pct')):
            summary_text += f"• {row['Model']:<12}: {row['Features_Kept_Pct']:.1f}% features kept\n"

summary_text += f"\n📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('polynomial_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save comparison results
summary_df.to_csv('polynomial_models_comparison.csv', index=False)

print_with_timestamp(
    "✅ Comparison visualization saved: polynomial_models_comparison.png")
print_with_timestamp(
    "✅ Comparison results saved: polynomial_models_comparison.csv")

# Print detailed recommendations
print_with_timestamp(f"\n🎯 RECOMMENDATIONS:")
print_with_timestamp("=" * 50)

best_model = summary_df.iloc[0]
print_with_timestamp(f"🏆 BEST OVERALL: {best_model['Model']}")
print_with_timestamp(f"   - Achieves {best_model['Best_Val_MAPE']:.2f}% MAPE")
print_with_timestamp(
    f"   - Training time: {best_model['Total_Training_Time']:.1f} seconds")

if len(summary_df) > 1:
    fastest_model = summary_df.loc[summary_df['Total_Training_Time'].idxmin()]
    if fastest_model['Model'] != best_model['Model']:
        print_with_timestamp(f"⚡ FASTEST: {fastest_model['Model']}")
        print_with_timestamp(
            f"   - Training time: {fastest_model['Total_Training_Time']:.1f} seconds")
        print_with_timestamp(
            f"   - Performance: {fastest_model['Best_Val_MAPE']:.2f}% MAPE")

# Model-specific recommendations
model_characteristics = {
    'Linear': "✨ Simple baseline, no regularization, fastest training",
    'Ridge': "🛡️ Good for correlated features, prevents overfitting via L2",
    'Lasso': "🎯 Automatic feature selection via L1, creates sparse models",
    'ElasticNet': "⚖️ Best of both worlds, combines L1 + L2 regularization"
}

print_with_timestamp(f"\n💡 Model Characteristics:")
for model_name in available_models.keys():
    if model_name in model_characteristics:
        print_with_timestamp(
            f"   {model_name}: {model_characteristics[model_name]}")

print_with_timestamp(f"\n🎉 Polynomial Regression Model Comparison Complete!")
print_with_timestamp(f"📁 Generated files:")
print_with_timestamp(
    f"   • polynomial_models_comparison.png - Visual comparison")
print_with_timestamp(
    f"   • polynomial_models_comparison.csv - Detailed metrics")
