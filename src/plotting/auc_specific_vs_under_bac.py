import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm
# File paths
# under_bac_csv = "/home/finkels9/parkinson/for_yoram_meeting/just_genus/lgr/picking_specific_bac_/under_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia;s__muciniphila_g__Faecalibacterium;s__prausnitzii.csv"
# specific_bac_csv = "/home/finkels9/parkinson/for_yoram_meeting/just_genus/lgr/picking_specific_bac_/specific_bac/g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia;s__muciniphila_g__Faecalibacterium;s__prausnitzii.csv"
under_bac_csv = r"C:\Users\user\PycharmProjects\pythonProject2\PPM-Across_Cohorts\save_results\genera\lgr\fair_selecting_based_training\under_bac\g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia_g__Faecalibacterium.csv"
specific_bac_csv = r"C:\Users\user\PycharmProjects\pythonProject2\PPM-Across_Cohorts\save_results\genera\lgr\fair_selecting_based_training\specific_bac\g__Roseburia_g__Bifidobacterium_g__Blautia_g__Lactobacillus_g__Akkermansia_g__Faecalibacterium.csv"
other_diseases=False
# Scenario mapping
scenario_mapping = {
    "Leave one dataset out; train on-16S": "16S/Shotgun",
    "Leave one dataset out; train on-Shotgun+16S": "Shotgun+16S",
    "Leave one dataset out; train on-Shotgun": "16S/Shotgun"
}
scenarios_of_interest = ["16S/Shotgun", "Shotgun+16S"]
named_scenario = ["16S/WGS", "16S+WGS"]

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

    if not other_diseases:
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
    'IBD', 'KIM', 'PRJEB47976_Alzheimer'
]

if other_diseases:
    mapping_project_to_nickname= {
        'PRJEB47976_Alzheimer': 'AD',
    'KIM': 'CRC',
    'IBD': 'IBD-2',
    'jacob': 'IBD-1',
    'HE': 'BF',
    }
else:
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
fig, ax = plt.subplots(figsize=(9, 9))

if not other_diseases:
    colors = {
        "Specific Bac": [ "#8575ff","#533dff"],
        "Under Bac": [ "#b80679", "#e835a9"]
    }
else:
    colors = {
        "Specific Bac": [ "#85d3f2","#007eb0"],
        "Under Bac": [ "#a188fc", "#6813e8"]
    }

# Plot data for Under Bac
scenario_data_one_under = under_bac_df[under_bac_df["Scenario"] == scenarios_of_interest[0]]
scenario_data_two_under = under_bac_df[under_bac_df["Scenario"] == scenarios_of_interest[1]]

based_type_under = pd.to_numeric(scenario_data_one_under["AUC"],errors='coerce').reset_index(drop=True)
all_under = pd.to_numeric(scenario_data_two_under["AUC"], errors='coerce').reset_index(drop=True)
ax.barh(
    y - bar_width / 2,
    based_type_under,
    bar_width,
    label=f"Under Taxa- {named_scenario[0]}",
    color=colors["Under Bac"][0],
    edgecolor='black',
    linewidth=1,
    zorder=3
)

ax.barh(
    y - bar_width / 2,
    all_under - based_type_under,
    bar_width,
    left=based_type_under,
    label=f"Under Taxa- {named_scenario[1]}",
    color=colors["Under Bac"][1]
)


# Plot data for Specific Bac
scenario_data_one_specific = specific_bac_df[specific_bac_df["Scenario"] == scenarios_of_interest[0]]
scenario_data_two_specific = specific_bac_df[specific_bac_df["Scenario"] == scenarios_of_interest[1]]

based_type_specific = pd.to_numeric(scenario_data_one_specific["AUC"],errors='coerce').reset_index(drop=True)
all_specific = pd.to_numeric(scenario_data_two_specific["AUC"],errors='coerce').reset_index(drop=True)

ax.barh(
    y + bar_width / 2,
    based_type_specific,
    bar_width,
    label=f"Specific Taxa- {named_scenario[0]}",
    color=colors["Specific Bac"][1],
    edgecolor='black',
    linewidth=1,
    zorder = 3

)

ax.barh(
    y + bar_width / 2,
    all_specific - based_type_specific,
    bar_width,
    left=based_type_specific,
    label=f"Specific Taxa- {named_scenario[1]}",
    color=colors["Specific Bac"][0],

)

average_based_type_under = based_type_under.mean()
average_all_under = all_under.mean()
average_based_type_specific = based_type_specific.mean()
average_all_specific = all_specific.mean()

# ax.axvline(0.67, color='red', linestyle='--', linewidth=2,
#            label=f'Average score PD-\n Under Taxa 16S+WGS (AUC={average_all_under:.2f})')

print(f"Average score of one type training, under taxa: {average_based_type_under}\n Average score of all types training, under taxa: {average_all_under}\n Average score of one type training,specific taxa {average_based_type_specific}\n Average score of all types training, specific taxa {average_all_specific}  ")
custom_font_12 = fm.FontProperties(family="DejaVu Serif", size=14)
custom_font_14 = fm.FontProperties(family="DejaVu Serif", size=14)

# Final adjustments
ax.set_yticks(y)
ax.set_yticklabels(projects, rotation=0, ha="right", fontproperties=custom_font_14)
ax.set_xlim(0.5, 1)
ax.set_xticks(np.arange(0.5, 1.0, 0.1))
ax.set_ylabel("Datasets", fontproperties=custom_font_12)
ax.set_xlabel("AUC", fontproperties=custom_font_12)
if other_diseases:
    ax.set_title("LODO Comparison- Similar Neurodegenerative and Other Diseases", fontproperties=custom_font_14)
else:
    ax.set_title("LODO Comparison- PD\nTaxa detected in the Training set ", fontproperties=custom_font_14)
legend_font = fm.FontProperties(family="DejaVu Serif", size=11)  # Adjust size if needed

#legend on top right
plt.legend(prop=legend_font,loc= 'upper right')

plt.tight_layout()
plt.savefig(r'C:\Users\user\PycharmProjects\pythonProject2\plots_to_figure\auc_pd',dpi=300)
plt.show()
