import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "scores/Meta-Llama-3-8B-Instruct/fancyzhx/ag_news/n56_target_description_and_classes_1/epoch_0006_acc_0.947000.csv"
df = pd.read_csv(csv_path)

# Pivot to grid: rows=source_layer, cols=target_layer, values=rouge1 (mean if duplicates)
grid = df.pivot_table(
    index="target_layer",
    columns="source_layer",
    values="rouge1",
    aggfunc="mean"
)

plt.figure(figsize=(10, 8))
sns.heatmap(grid, cmap="viridis", annot=False)
plt.title("ROUGE1 Smoothness by Src / Tgt Layer")
plt.xlabel("Src Layer")
plt.ylabel("Tgt Layer")
plt.tight_layout()
plt.savefig('viz/heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
