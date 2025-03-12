import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example DataFrame
path = "/home/finkels9/parkinson/NEW_RESULTS/lgr/picking_specific_bac_/under_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia_g__Faecalibacterium.csv"

df = pd.read_csv(path, index_col=0)
filtered_rows = df[df['Scenario'] == 'Leave one dataset out; train on-Shotgun+16S']
df = filtered_rows.loc['jacob']
coefficients_str = df['Coefficients']
bacteria_coefficients = [
    item.split(':') for item in coefficients_str.split(', ')
]

# Extract bacteria names and coefficients
bacteria_names = [bc[0] for bc in bacteria_coefficients]
coefficients = [float(bc[1]) * 100 for bc in bacteria_coefficients]
group = [name.split('g__')[1].split(';')[0].split('_')[0] if 'g__' in name else None for name in bacteria_names]
names = [name.split('g__')[1].split(';')[1] if 'g__' in name else None for name in bacteria_names]

df = pd.DataFrame({
    'group': group,
    'name': names,
    'value': coefficients
})

# Reorder the DataFrame
df_sorted = (
    df.groupby("group")
    .apply(lambda x: x.sort_values("value", ascending=False))
    .reset_index(drop=True)
)

VALUES = df_sorted["value"].values
LABELS = df_sorted["name"].values
GROUP = df_sorted["group"].values

PAD = 2
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]

offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Set up polar plot
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "polar"})
ax.set_theta_offset(np.pi / 2)  # Adjust starting position of the plot
ax.set_ylim(-100, 100)  # Define limits for the radius
ax.set_frame_on(False)  # Remove polar frame
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Generate colors
unique_groups = np.unique(GROUP)
color_map = plt.cm.get_cmap("tab20", len(unique_groups))
group_colors = {group: color_map(i) for i, group in enumerate(unique_groups)}
COLORS = [group_colors[g] for g in GROUP]
# Plot bars
ax.bar(
    ANGLES[IDXS],
    VALUES,
    width=WIDTH,
    color=COLORS,
    edgecolor="white",
    linewidth=1
)


# Add labels
def add_labels(angles, values, labels, offset, ax):
    for angle, value, label in zip(angles, values, labels):
        alignment = "center"
        rotation = np.degrees(angle) - 90 if angle < np.pi else np.degrees(angle) + 90
        ax.text(
            angle,
            value + offset if value >= 0 else value - offset,
            label,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=10
        )

# Add group names
group_angles = [np.mean(ANGLES[IDXS][GROUP == g]) for g in unique_groups]
for g, angle in zip(unique_groups, group_angles):
    ax.text(
        angle,
        110,  # Position outside the circle
        g,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=group_colors[g]
    )

plt.show()
