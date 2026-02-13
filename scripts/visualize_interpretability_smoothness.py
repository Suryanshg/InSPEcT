import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Define the folder containing CSV files
csv_folder = Path("cot_scores/Meta-Llama-3-8B-Instruct/SetFit/subj/n56_target_description_and_classes_1")

# Read all CSV files and concatenate
all_dfs = []
for csv_file in csv_folder.glob("*.csv"):
    if str(csv_file).__contains__("epoch"):
        df = pd.read_csv(csv_file)
        print(f"\n{csv_file.name}: {len(df)} rows, rouge1 range: {df['rouge1'].min():.3f} - {df['rouge1'].max():.3f}")
        all_dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)

# Pivot to grid
grid = combined_df.pivot_table(
    index="target_layer",
    columns="source_layer",
    values="rouge1",
    aggfunc="mean"
)

plt.figure(figsize=(10, 8))
sns.heatmap(grid, cmap="viridis", annot=False)
plt.title(f"Subj 56 Tokens - ROUGE1 Smoothness by Src / Tgt Layer (Mean of {len(all_dfs)} inference runs)")
plt.xlabel("Src Layer")
plt.ylabel("Tgt Layer")
plt.tight_layout()
plt.savefig('viz/heatmap.png', dpi=150, bbox_inches='tight')
plt.show()