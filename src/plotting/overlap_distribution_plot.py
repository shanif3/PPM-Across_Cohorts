import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import seaborn as sns
# Modify legend to show only two categories
from matplotlib.patches import Patch
# Load data
all_data = pd.read_pickle("/home/finkels9/parkinson/projects_mimic_results_peptibase.pkl")
datasets_16S = []
datasets_WGS = []

for key in all_data.keys():
    if key != 'processed_all':
        tag, path_to_read, df_corrs, processed = all_data[key]['tag'], all_data[key]['path_to_read'], all_data[key]['df_corrs'], all_data[key]['processed']
        u_col = [col for col in processed.columns if len(processed[col].unique()) > 1]
        if '16S' in path_to_read:
            # get the columns that has at least 2 unique values
            datasets_16S.extend(u_col)
        if 'Shotgun' in path_to_read:
            datasets_WGS.extend(u_col)

# Extract leaf genus level (Species-Level under Genus)
just_leaf_genus_16S = [x for x in datasets_16S if x.split(';')[5] != 'g__' and x.split(';')[6] == 's__']
just_leaf_genus_WGS = [x for x in datasets_WGS if x.split(';')[5] != 'g__' and x.split(';')[6] == 's__']
all_leaf_genus = just_leaf_genus_16S + just_leaf_genus_WGS

# Count occurrences
unique_16S = Counter(datasets_16S)
unique_WGS = Counter(datasets_WGS)
unique_all = Counter(datasets_16S + datasets_WGS)

genus_leaf_unique_16S = Counter(just_leaf_genus_16S)
genus_leaf_unique_WGS = Counter(just_leaf_genus_WGS)
genus_leaf_unique_all = Counter(all_leaf_genus)

# Extract counts
counts_16S = list(unique_16S.values())
counts_WGS = list(unique_WGS.values())
counts_all = list(unique_all.values())

genus_leaf_counts_16S = list(genus_leaf_unique_16S.values())
genus_leaf_counts_WGS = list(genus_leaf_unique_WGS.values())
genus_leaf_counts_all = list(genus_leaf_unique_all.values())

# Fix DataFrame alignment
df = pd.DataFrame({
    "Occurrences": counts_16S + counts_WGS + counts_all + genus_leaf_counts_16S + genus_leaf_counts_WGS + genus_leaf_counts_all,
    "Category": (["Leaf"] * (len(counts_16S) + len(counts_WGS) + len(counts_all))) +
                (["Genus"] * (len(genus_leaf_counts_16S) + len(genus_leaf_counts_WGS) + len(genus_leaf_counts_all))),
    # "Type": (["16S Leaf"] * len(counts_16S)) +
    #         (["WGS Leaf"] * len(counts_WGS)) +
    #         (["Combined Leaf"] * len(counts_all)) +
    #         (["16S Genus"] * len(genus_leaf_counts_16S)) +
    #         (["WGS Genus"] * len(genus_leaf_counts_WGS)) +
    #         (["Combined Genus"] * len(genus_leaf_counts_all))
})
# Define custom colors
custom_palette = {
    "Leaf": "#fdc2ff",  # Red-Orange
    "Leaf of WGS": "#fdc2ff",  # Green
    "Leaf of 16S & WGS": "#fdc2ff",  # Blue
    "Genus": "#7b00ff",  # Pink
    "Genus Leaf of WGS": "#7b00ff",  # Purple
    "Genus Leaf of 16S & WGS": "#7b00ff"  # Cyan
}
plt.figure(figsize=(4, 4))  # Make the plot smaller

for t in df["Category"].unique():
    subset = df[df["Category"] == t]
    sns.kdeplot(subset["Occurrences"], fill=True, color=custom_palette[subset["Category"].iloc[0]])


legend_elements = [
    Patch(facecolor=custom_palette["Leaf"], label="Leaf", alpha=0.6),
    Patch(facecolor=custom_palette["Genus"], label="Genus", alpha=0.6)
]
plt.legend(handles=legend_elements, loc="upper right")
plt.xlim(-1, 10)
plt.title("Leaf vs Genus Occurrences Across Datasets")
plt.xlabel("Number of Occurrences Across Datasets")
plt.ylabel("Density")
plt.show()
