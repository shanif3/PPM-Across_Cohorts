import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
# under_bac_csv = "/home/finkels9/parkinson/for_yoram_meeting/just_genus/lgr/picking_specific_bac_/under_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia;s__muciniphila_g__Faecalibacterium;s__prausnitzii.csv"
# specific_bac_csv = "/home/finkels9/parkinson/for_yoram_meeting/just_genus/lgr/picking_specific_bac_/specific_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia;s__muciniphila_g__Faecalibacterium;s__prausnitzii.csv"
under_bac_csv = "/home/finkels9/parkinson/for_yoram_meeting/just_genus/lgr/fair_selecting_based_training/under_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia_g__Faecalibacterium.csv"
specific_bac_csv = "/home/finkels9/parkinson/for_yoram_meeting/just_genus/lgr/fair_selecting_based_training/specific_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia_g__Faecalibacterium.csv"

# Scenario mapping
scenario_mapping = {
    "Leave one dataset out; train on-16S": "16S/Shotgun",
    "Leave one dataset out; train on-Shotgun+16S": "Shotgun+16S",
    "Leave one dataset out; train on-Shotgun": "16S/Shotgun"
}
scenarios_of_interest = ["16S/Shotgun", "Shotgun+16S"]

# Function to preprocess CSV
def preprocess_csv(file_path, scenario_mapping, scenarios_of_interest, project_by_order, mapping_project_to_nickname):
    df = pd.read_csv(file_path)
    df['Scenario'] = df['Scenario'].map(scenario_mapping)
    df = df[df['Scenario'].isin(scenarios_of_interest)]
    df['name'] = df['Project'] + df['Scenario']

    df = df.set_index('Project').loc[project_by_order].reset_index()
    df = df[df['Project'].isin(mapping_project_to_nickname.keys())]
    df['Project'] = df['Project'].map(mapping_project_to_nickname)
    df['p'] = df['Project'] + df['Scenario']

    cols_of_Validation= [col for col in df['p'] if col.startswith('Validation')]
    columns_without_validation = [col for col in df['p'] if not col.startswith('Validation')]
    reordered_columns = sorted(columns_without_validation, key=lambda x: (x.split('-')[1], int(x.split('-')[0])))
    reordered_columns= reordered_columns+ cols_of_Validation
    reordered_columns= reversed(reordered_columns)
    df = df.set_index('p').loc[reordered_columns].reset_index()
    return df

# Project order and mapping
project_by_order = [
    'PRJNA381395', 'PRJNA1101026', 'PRJEB27564_baseline', 'PRJEB30615', 'PRJEB14674',
    'PRJNA601994_dataset2', 'PRJEB27564_followup', 'PRJNA494620', 'PRJNA510730', 'PRJNA762484',
    'PRJNA743718', 'PRJEB17784', 'PRJEB53401', 'PRJEB53403', 'PRJEB59350', 'Linoy', 'HE', 'jacob',
    'IBD', 'KIM', 'CIRRHOSIS', 'PRJEB47976_Alzheimer'
]

# mapping_project_to_nickname = {
#     'HE': '1-Val-16S',
#     'jacob': '2-Val-16S',
#     'IBD': '3-Val-16S',
#     'KIM': '4-Val-16S',
#     'PRJEB47976_Alzheimer': '6-Val-WGS'
# }

# mapping_project_to_nickname= {
#     'PRJEB47976_Alzheimer': 'AD',
# 'KIM': 'CRC',
# 'IBD': 'IBD-2',
# 'jacob': 'IBD-1',
# 'HE': 'BF',
# }

mapping_project_to_nickname={
    'PRJEB27564_baseline': '1-16S',
    'PRJEB27564_followup': '2-16S',
    'PRJEB14674': '3-16S',
    'PRJNA510730': '4-16S',
    'PRJEB14928': '5-16S',
    'PRJNA494620': '6-16S',
    'PRJEB30615': '7-16S',
    'PRJNA601994_dataset2': '8-16S',
    'PRJNA1101026': '9-16S',
    'PRJNA381395': '10-16S',
    'PRJEB53403':'1-WGS',
    'PRJNA743718':'2-WGS',
    'PRJEB17784':'3-WGS',
    'PRJEB53401':'4-WGS',
    'PRJEB59350':'5-WGS',
    'PRJNA762484':'6-WGS',
    'Linoy':'Validation'
}

# Preprocess both CSVs
under_bac_df = preprocess_csv(under_bac_csv, scenario_mapping, scenarios_of_interest, project_by_order, mapping_project_to_nickname)
specific_bac_df = preprocess_csv(specific_bac_csv, scenario_mapping, scenarios_of_interest, project_by_order, mapping_project_to_nickname)

# Separate data by scenarios
scenarios = under_bac_df["Scenario"].unique()
projects = under_bac_df["Project"].unique()

# Prepare data for plotting
y = np.arange(len(projects))  # y positions for the projects
bar_width = 0.25  # Width of each bar

# Initialize the plot
fig, ax = plt.subplots(figsize=(6, 6))

colors = {
    "Specific Bac": ["#533dff", "#3f2fc2"],
    "Under Bac": [ "#b80679", "#e835a9"]
}

# Plot data for Under Bac
scenario_data_one_under = under_bac_df[under_bac_df["Scenario"] == scenarios_of_interest[0]]
scenario_data_two_under = under_bac_df[under_bac_df["Scenario"] == scenarios_of_interest[1]]

based_type_under = scenario_data_one_under["AUC"].reset_index(drop=True)
all_under = scenario_data_two_under["AUC"].reset_index(drop=True)

ax.barh(
    y - bar_width / 2,
    based_type_under,
    bar_width,
    label=f"Under Bac {scenarios_of_interest[0]}",
    color=colors["Under Bac"][0]
)

ax.barh(
    y - bar_width / 2,
    all_under - based_type_under,
    bar_width,
    left=based_type_under,
    label=f"Under Bac {scenarios_of_interest[1]}",
    color=colors["Under Bac"][1]
)


# Plot data for Specific Bac
scenario_data_one_specific = specific_bac_df[specific_bac_df["Scenario"] == scenarios_of_interest[0]]
scenario_data_two_specific = specific_bac_df[specific_bac_df["Scenario"] == scenarios_of_interest[1]]

based_type_specific = scenario_data_one_specific["AUC"].reset_index(drop=True)
all_specific = scenario_data_two_specific["AUC"].reset_index(drop=True)

ax.barh(
    y + bar_width / 2,
    based_type_specific,
    bar_width,
    label=f"Specific Bac {scenarios_of_interest[0]}",
    color=colors["Specific Bac"][1]
)

ax.barh(
    y + bar_width / 2,
    all_specific - based_type_specific,
    bar_width,
    left=based_type_specific,
    label=f"Specific Bac {scenarios_of_interest[1]}",
    color=colors["Specific Bac"][0]
)

# Final adjustments
ax.set_yticks(y)
ax.set_yticklabels(projects, rotation=0, ha="right")
ax.set_xlim(0.5, 1)
ax.set_xticks(np.arange(0.5, 1.0, 0.1))
ax.set_ylabel("Projects", fontsize=12)
ax.set_xlabel("AUC", fontsize=12)
ax.set_title("LODO Comparison- Similar neurodegenerative diseases- ", fontsize=14)
# ax.set_title("LODO Comparison- PD", fontsize=14)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
