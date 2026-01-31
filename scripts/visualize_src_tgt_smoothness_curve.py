import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "scores/Meta-Llama-3-8B-Instruct/stanfordnlp/sst2/n7_target_description_and_classes_1/epoch_0008_acc_0.897936.csv"
df = pd.read_csv(csv_path)

# Filter out rows where rouge1 AND class_rate is > 0
df_filtered = df[(df['rouge1'] > 0) & (df['class_rate'] > 0)]

# Create a combined label for each source-target pair
df_filtered = df_filtered.copy()
df_filtered['pair'] = df_filtered['source_layer'].astype(str) + '→' + df_filtered['target_layer'].astype(str)
df_filtered = df_filtered.sort_values(['source_layer', 'target_layer'])

# Create figure with single plot and two y-axes
fig, ax1 = plt.subplots(figsize=(16, 6))

# ROUGE-1 line plot (left y-axis)
color1 = 'steelblue'
ax1.plot(range(len(df_filtered)), df_filtered['rouge1'], marker='o', color=color1, alpha=0.8, linewidth=1.5, markersize=4, label='ROUGE-1')
ax1.set_xticks(range(len(df_filtered)))
ax1.set_xticklabels(df_filtered['pair'], rotation=90, fontsize=7)
ax1.set_xlabel('Source → Target Layer')
ax1.set_ylabel('ROUGE-1 Score')
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)

# Class Rate line plot (right y-axis)
ax2 = ax1.twinx()
color2 = 'darkorange'
ax2.plot(range(len(df_filtered)), df_filtered['class_rate'], marker='s', color=color2, alpha=0.8, linewidth=1.5, markersize=4, label='Class Rate')
ax2.set_ylabel('Class Rate')
ax2.tick_params(axis='y')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('ROUGE-1 Score (>0) and Class Rate (>0) by Src - Tgt Layer Pairing')
plt.tight_layout()
plt.savefig('layer_scores.png', dpi=150, bbox_inches='tight')
plt.show()