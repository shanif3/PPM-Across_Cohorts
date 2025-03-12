import numpy as np
from matplotlib.patches import Patch
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
mapping_project_to_nickname={
    'PRJEB27564_baseline': '1-16S',
    'PRJEB27564_followup': '2-16S',
    'PRJEB14674': '3-16S',
    'PRJNA510730': '4-16S',
    'PRJNA494620': '5-16S',
    'PRJEB30615': '6-16S',
    'PRJNA601994_dataset2': '7-16S',
    'PRJNA1101026': '8-16S',
    'PRJNA381395': '9-16S',
    'PRJEB53403':'1-WGS',
    'PRJNA743718':'2-WGS',
    'PRJEB17784':'3-WGS',
    'PRJEB53401':'4-WGS',
    'PRJEB59350':'5-WGS',
    'PRJNA762484':'6-WGS'
}
# Example data: Dictionary of datasets and their significant microbes
projects = ['PRJEB14674', 'PRJEB14928', 'PRJEB27564_baseline', 'PRJEB27564_followup', 'PRJEB30615', 'PRJNA381395',
            'PRJNA494620',
            'PRJNA601994_dataset2', 'PRJNA1101026', 'PRJNA510730', 'PRJEB17784', 'PRJEB53401', 'PRJEB53403',
            'PRJNA743718', 'PRJEB59350', 'PRJNA762484']
data = {project: {} for project in projects}

all_projects = pd.read_pickle('projects_mimic_results_peptibase.pkl')
projects = [p for p in all_projects.keys() if p in projects]

for project in projects:
    df_corrs = all_projects[project]['df_corrs']
    if not isinstance(df_corrs, pd.DataFrame):
        continue
    data[project]['len'] = len(df_corrs)
    for bac, row_data in df_corrs.iterrows():
        coeff = float(row_data['scc'])  # Extract and convert the coeff value to float

        # Split and process the 'bac' column
        bact_split = bac.split(';')[-3:]
        if len(bact_split) == 1:
            continue

        for index, bact_sp in enumerate(bact_split):
            if bact_sp.startswith('g__'):
                bact_split = bact_split[index:]
                new_bac = bact_split
                if len(bact_split) == 1:
                    genus = 'g__' + bact_split[0].split('__')[1].split('_')[0]
                    data[project][genus] = coeff
                else:
                    a = bact_split[1].split('__')[1].split('_')[0]
                    check_if_specie_has_genus_name = 'g__' + f'{a}'
                    if bact_split[0] == check_if_specie_has_genus_name:
                        new_bac = bact_split[0] + ';' + bact_split[1].replace(
                            bact_split[1].split('__')[1].split('_')[0] + '_',
                            ''
                        )
                        new_bac = new_bac.split(';')

                    if len(new_bac) == 1:
                        genus = new_bac[0]
                        data[project][genus] = coeff
                    elif len(new_bac) == 3:
                        genus = ';'.join(new_bac)
                        data[project][genus] = coeff
                    elif len(new_bac) == 2 and bact_split[-1].startswith('t__'):
                        genus = ';'.join(new_bac) + f';{bact_split[-1]}'
                        data[project][genus] = coeff
                    else:
                        genus = ';'.join(new_bac)
                        data[project][genus] = coeff
            else:
                continue
            break

bacteria_counter = Counter()
for microbes in data.values():
    bacteria_counter.update(microbes.keys())
    # bacteria_counter.update(micro.split(';')[0] for micro in microbes)

filtered_bacteria = {bacteria for bacteria, count in bacteria_counter.items() if count>2}
filtered_projects = {project: set(microbes.keys()).intersection(filtered_bacteria) for project, microbes in
                     data.items()}
# bac= []
# for project, bacteria_list in data.items():
#     bac.extend(set(bacteria_list))
# bacteria_counter= Counter(bac)
# filtered_bacteria= [bacteria for bacteria, count in bacteria_counter.items() if count >=3]
# filtered_projects = {project: set(microbes.keys()).intersection(filtered_bacteria) for project, microbes in
#                      data.items()}


def create_presence_matrix_with_scc(filtered_projects, data):
    """
    Create a binary matrix indicating the presence of bacteria across datasets and include SCC values.

    Parameters:
        filtered_projects (dict):
            A dictionary where keys are dataset names and values are sets of bacteria.
        data (dict):
            A dictionary containing the SCC value for each bacterium in each dataset.

    Returns:
        pd.DataFrame, pd.DataFrame:
            A binary presence matrix and an SCC matrix.
    """
    # remove 'len' key from filtered_projects
    for project in filtered_projects:
        filtered_projects[project].discard('len')
    all_bacteria = sorted(set.union(*filtered_projects.values()))

    presence_matrix = pd.DataFrame(0, index=all_bacteria, columns=filtered_projects.keys())
    scc_matrix = pd.DataFrame(0.0, index=all_bacteria, columns=filtered_projects.keys())

    for project, bacterias in filtered_projects.items():
        for bacteria in bacterias:
            if bacteria != 'len':
                presence_matrix.loc[bacteria, project] = 1
                scc_value = data[project].get(bacteria, 0)
                scc_matrix.loc[bacteria, project] = scc_value

    presence_matrix = presence_matrix.loc[:, (presence_matrix != 0).any(axis=0)]
    scc_matrix = scc_matrix.loc[:, (scc_matrix != 0).any(axis=0)]

    return presence_matrix, scc_matrix


def plot_bacteria_presence_with_scc(presence_matrix, scc_matrix, data):
    """
    Create an UpSet-like presence plot of bacteria across datasets with SCC-based coloring and dot sizes based on SCC magnitude.

    Parameters:
        presence_matrix (pd.DataFrame):
            A dataframe where columns are dataset numbers and the index are bacterial names.
            The values indicate the presence (1) or absence (0) of a bacterium in the dataset.
        scc_matrix (pd.DataFrame):
            A dataframe where columns are dataset numbers and the index are bacterial names.
            The values indicate the SCC for a bacterium in the dataset.
    """
    binary_matrix = (presence_matrix > 0).astype(int).T


    scc_matrix = scc_matrix.T

    col_sums = binary_matrix.sum(axis=0)


    scc_matrix = scc_matrix.loc[binary_matrix.index, binary_matrix.columns]
    binary_matrix = binary_matrix[col_sums.sort_values(ascending=False).index]

    fig = plt.figure(figsize=(38, 28))
    grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.1, wspace=0.1)

    # Top barplot
    ax_col_bar = fig.add_subplot(grid[0, 0])
    col_sums = col_sums.loc[binary_matrix.columns]
    bar_container = ax_col_bar.bar(binary_matrix.columns, col_sums, color='#bba3ff')  # Pastel pink
    ax_col_bar.set_xticks([])
    ax_col_bar.set_ylabel('Bacteria overlap count', fontsize=30)
    ax_col_bar.spines['top'].set_visible(False)
    ax_col_bar.spines['right'].set_visible(False)
    ax_col_bar.spines['left'].set_visible(False)
    ax_col_bar.bar_label(bar_container, labels=[f'{int(c)}' for c in col_sums], padding=3, fontsize=40)

    # Left barplot
    ax_row_bar = fig.add_subplot(grid[1, 1])
    project_counts = {}
    nickname_to_project = {v: k for k, v in mapping_project_to_nickname.items()}
    for project in binary_matrix.index:
        project1= nickname_to_project.get(project)
        project_counts[project] = data[project1]['len']

    project_counts = pd.Series(project_counts)
    project_counts = project_counts.loc[binary_matrix.index]
    barh_container = ax_row_bar.barh(project_counts.index, project_counts, color='#bba3ff')
    ax_row_bar.set_yticks([])
    # add the project counts to the right of the bar
    ax_row_bar.set_xlim(0, project_counts.max() + 1,)
    ax_row_bar.set_xlabel('Significant bacteria count on project', fontsize=30)
    ax_row_bar.spines['top'].set_visible(False)
    ax_row_bar.spines['right'].set_visible(False)
    ax_row_bar.spines['bottom'].set_visible(False)
    ax_row_bar.bar_label(barh_container, labels=[f'{int(c)}' for c in project_counts], padding=3, fontsize=40)



    ax_grid = fig.add_subplot(grid[1, 0])
    ax_grid.set_xlim(-0.5, len(binary_matrix.columns) - 0.5)
    ax_grid.set_ylim(-0.5, len(binary_matrix.index) - 0.5)
    ax_grid.set_facecolor('#F5F5F5')

    # normalize scc values by project
    for project in binary_matrix.index:
        scc_matrix.loc[project] = scc_matrix.loc[project] / scc_matrix.loc[project].abs().max()
    for i, (dataset, row) in enumerate(binary_matrix.iterrows()):
        for j, value in enumerate(row):
            scc_value = scc_matrix.loc[dataset, binary_matrix.columns[j]]
            color = '#0f60f7' if scc_value > 0 else '#db1435'
            size = 20 + abs(scc_value) * 40  # Base size 10, scaled by SCC magnitude
            if value == 1:
                ax_grid.plot(j, i, marker='o', color=color, markersize=size, markeredgecolor='black', markeredgewidth=1)
            else:
                ax_grid.plot(j, i, marker='o', color='#E6E6FA', markersize=20, markeredgecolor='#BBBBBB',
                             markeredgewidth=0.5)

    ax_grid.set_xticks(np.arange(len(binary_matrix.columns)))
    corrected_binary_matrix_columns = [col.rsplit('_', 1)[0] if len(col.split(';')) > 1 else col for col in
                                       binary_matrix.columns]

    ax_grid.set_xticklabels(corrected_binary_matrix_columns, rotation=18, fontsize=35)
    ax_grid.set_yticks(np.arange(len(binary_matrix.index)))
    ax_grid.set_yticklabels(binary_matrix.index, fontsize=60)

    legend_elements = [
        Patch(facecolor='#0f60f7', label='Significant & Positive Coefficient'),
        Patch(facecolor='#db1435', label='Significant & Negative Coefficient'),
        Patch(facecolor='#E6E6FA', label='Not Significant')
    ]
    ax_grid.legend(handles=legend_elements, loc='lower right', fontsize=35)
    ax_grid.set_facecolor('white')  # Set background color to white
    # Remove the border (spines) of the grid
    ax_grid.spines['top'].set_visible(False)
    ax_grid.spines['right'].set_visible(False)
    ax_grid.spines['bottom'].set_visible(False)
    ax_grid.spines['left'].set_visible(False)

    # Remove grid ticks for a clean look
    ax_grid.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    plt.show()


presence_matrix, scc_matrix = create_presence_matrix_with_scc(filtered_projects, data)
presence_matrix.columns= [mapping_project_to_nickname[i] for i in presence_matrix.columns]
# sort the columns
reordered_columns = sorted(presence_matrix.columns, key=lambda x: (x.split('-')[1], int(x.split('-')[0])))

presence_matrix = presence_matrix[reordered_columns]
scc_matrix.columns= [mapping_project_to_nickname[i] for i in scc_matrix.columns]
scc_matrix = scc_matrix[reordered_columns]
presence_matrix=presence_matrix[presence_matrix.columns[::-1]]
scc_matrix=scc_matrix[scc_matrix.columns[::-1]]

plot_bacteria_presence_with_scc(presence_matrix, scc_matrix, data)
