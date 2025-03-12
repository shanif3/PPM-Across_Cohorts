import os
from collections import Counter
from pathlib import Path

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from src.miMic.mimic_test import apply_mimic


def fix_substring(substrings):
    """Fixes substrings by appending taxonomy levels."""
    fixed_substrings = []
    base_levels = ['p__', 'c__', 'o__', 'f__', 'g__', 's__', 't__']

    for sub in substrings:
        last_level = sub.split(';')[-1].split('__')[0]
        index = base_levels.index(last_level + '__')
        fixed = sub + ';' + ';'.join(base_levels[index + 1:])
        fixed_substrings.append(fixed)
        fixed = sub + '_0' + ';' + ';'.join(base_levels[index + 1:])
        fixed_substrings.append(fixed)
    return fixed_substrings


def overlap_check(projects_number, project_mimic_result):
    """
    Checks for overlapping bacteria across multiple projects and filters them based on occurrence.

    Args:
        projects_number (list): List of project identifiers.
        project_mimic_result (dict): Dictionary containing project data, including correlations.

    Returns:
        set: A set of bacteria that overlap across projects more than twice.
    """

    data = {project: {} for project in projects_number}

    for project in projects_number:
        df_corrs = project_mimic_result.get(project, {}).get('df_corrs')
        if df_corrs is None:
            continue

        for bac, row in df_corrs.iterrows():
            coeff = float(row['scc'])  # Replace with the actual column name in df_corrs
            bact_split = bac.split(';')[-3:]

            if len(bact_split) == 1:
                continue

            for index, bact_sp in enumerate(bact_split):
                if bact_sp.startswith('g__'):
                    bact_split = bact_split[index:]

                    if len(bact_split) == 1:
                        genus = 'g__' + bact_split[0].split('__')[1].split('_')[0]
                        data[project][genus] = coeff
                    else:
                        genus_name = bact_split[1].split('__')[1].split('_')[0]
                        check_genus = 'g__' + genus_name

                        if bact_split[0] == check_genus:
                            bact_split[1] = bact_split[1].replace(genus_name + '_', '')

                        new_bac = ';'.join(bact_split)

                        if len(bact_split) == 3:
                            data[project][new_bac] = coeff
                        elif len(bact_split) == 2 and bact_split[-1].startswith('t__'):
                            data[project][new_bac] = coeff
                        else:
                            data[project][new_bac] = coeff

                    break

    # Count occurrences of each bacterium across projects
    bacteria_counter = Counter()
    for microbes in data.values():
        bacteria_counter.update(microbes.keys())

    # Filter bacteria that occur in more than two projects
    filtered_bacteria = {bacteria for bacteria, count in bacteria_counter.items() if count > 2}

    return filtered_bacteria


def save_coefficients_to_csv(mean_coefficients, feature_names, filename):
    df = pd.DataFrame({
        'Name': feature_names,
        'Coefficient': mean_coefficients
    })
    df.to_csv(filename, index=False)


def plot_roc(path_to_save, train_on, pred_probabilities, true_labels, loo, lodo, project_lodo, validation,
             confounders_names_list,
             strain_flag=False):
    if confounders_names_list == None:
        name = f"{project_lodo}_{train_on}"
        name = re.sub(r'[\\/]', '_', name)

    else:
        name = f"{project_lodo}_{train_on}_confounders"
    if loo == '_llo':
        type_name = f'Leave one out- {name}'
        extend_name = ''
    elif lodo:
        train_on_filter = " ".join(train_on.split('_'))
        if validation:
            type_name = f'test= validation {project_lodo}, train= rest of {train_on}'
        else:
            type_name = f'test= {project_lodo}, train= rest of {train_on_filter}'

        if confounders_names_list == None:
            extend_name = f'all_{train_on}_without_{project_lodo}_'
        else:
            confounders_names = '_'.join(confounders_names_list)
            extend_name = f'confounders_{confounders_names}_all_{train_on}_without_{project_lodo}_'
        name = ''
    else:
        type_name = f'{name}'
        extend_name = ''

    fpr, tpr, _ = roc_curve(true_labels, pred_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title(f'{type_name}', fontsize=15, fontweight="bold")
    plt.legend(loc='lower right')
    plt.savefig(f'{path_to_save}/{extend_name}{name}_roc_curve{loo}.png')

    if project_lodo  =='Linoy':
        pd.to_pickle([fpr, tpr], f'{path_to_save}/{extend_name}{name}_roc_curve{loo}_values_fpr_tpr.pkl')


def fix_and_cut_in_genara(relevant_bac):
    new_rel_bac = []

    for bac in relevant_bac:
        if bac == 'k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Betaproteobacteriales;f__Burkholderiaceae;g__Sutterella;Ambiguous_taxa_0;t__':
            bac = 'k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Betaproteobacteriales;f__Burkholderiaceae;g__Sutterella;s__Ambiguous_taxa_0;t__'

        if bac =='k__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Akkermansiaceae;g__Akkermansia;Ambiguous_taxa_0;t__':
            bac = 'k__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Akkermansiaceae;g__Akkermansia;s__Ambiguous_taxa_0;t__'

        bact_split = bac.split(';')[-3:]
        if len(bact_split) == 1:
            continue
        for index, bact_sp in enumerate(bact_split):
            if bact_sp.startswith('g__'):
                bact_split = bact_split[index:]
                new_bac = bact_split
                if len(bact_split) == 1:
                    new_rel_bac.append('g__' + bact_split[0].split('__')[1].split('_')[0])
                else:
                    first_name_of_specie = bact_split[1].split('__')[1].split('_')[0]
                    check_if_specie_has_genus_name = 'g__' + f'{first_name_of_specie}'
                    if bact_split[0] == check_if_specie_has_genus_name:
                        new_bac = bact_split[0] + ';' + bact_split[1].replace(
                            bact_split[1].split('__')[1].split('_')[0] + '_',
                            '')
                        new_bac = new_bac.split(';')

                    elif len(new_bac) == 1:
                        new_rel_bac.append(new_bac)
                    if len(new_bac) == 3:
                        new_rel_bac.append(';'.join(new_bac))
                    elif len(new_bac) == 2 and bact_split[-1].startswith('t__'):
                        new_rel_bac.append(';'.join(new_bac) + f';{bact_split[-1]}')
                    else:
                        new_rel_bac.append(';'.join(new_bac))

            else:
                continue
            break

    return new_rel_bac


def get_name_to_use(leave_one_out_flag, leave_one_dataset_out_flag, confounders_names_list, project_number, train_on,
                    validation):
    train_on = 'all_16S_and_shotgun' if type == 'all' else train_on

    project_number = re.sub(r'[\\/]', '_', project_number)
    if leave_one_out_flag:
        loo = '_llo'
    elif not leave_one_out_flag and not leave_one_dataset_out_flag:
        loo = ''
    if confounders_names_list != None:
        extend_name = 'cofounders_'
    else:
        extend_name = ''
    if leave_one_dataset_out_flag:
        loo = ''
    extend_name = f'{extend_name}all_{train_on}_without_{project_number}_'


    return extend_name, loo


def get_subdirectories(root_path):
    root_path = Path(root_path)
    result = {}
    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            # Get only directory names in the current subdirectory
            result[sub_dir.name] = [d.name for d in sub_dir.iterdir() if d.is_dir()]
    return result


def check_required_directories(root_path):
    required_dirs = ["16S", "Shotgun", "validation"]
    required_files = ["for_preprocess.csv", "tag.csv"]
    validation_required_dirs = ["16S", "Shotgun"]

    existing_dirs = {d: os.path.join(root_path, d) for d in os.listdir(root_path) if
                     os.path.isdir(os.path.join(root_path, d))}
    missing_dirs = [d for d in required_dirs if d not in existing_dirs]
    extra_dirs = [d for d in existing_dirs if d not in required_dirs]

    if missing_dirs or extra_dirs:
        raise ValueError(f"Missing directories: {missing_dirs}, Extra directories: {extra_dirs}")

    # Validate files in each subdirectory
    file_validation = {}
    all_subdirs = []
    for main_dir, main_dir_path in existing_dirs.items():
        if main_dir == 'validation':
            continue
        sub_dirs = [os.path.join(main_dir_path, sub) for sub in os.listdir(main_dir_path) if
                    os.path.isdir(os.path.join(main_dir_path, sub))]
        all_subdirs.extend(sub_dirs)
        file_validation[main_dir] = {}

        for sub_dir in sub_dirs:
            sub_dir_name = os.path.basename(sub_dir)
            missing_files = [file for file in required_files if not os.path.exists(os.path.join(sub_dir, file))]

            file_validation[main_dir][sub_dir_name] = {
                "all_files_present": not missing_files,
                "missing_files": missing_files
            }

    validation_path = existing_dirs["validation"]
    validation_sub_dirs = {d: os.path.join(validation_path, d) for d in os.listdir(validation_path) if
                           os.path.isdir(os.path.join(validation_path, d))}
    # check if the reqired subdirectories are present
    missing_dirs = [d for d in validation_required_dirs if d not in validation_sub_dirs]
    extra_dirs = [d for d in validation_sub_dirs if d not in validation_required_dirs]

    if missing_dirs or extra_dirs:
        raise ValueError(f"Missing directories: {missing_dirs}, Extra directories: {extra_dirs}")

    for val_main_dir, val_main_dir_path in validation_sub_dirs.items():
        deeper_sub_dirs = [os.path.join(val_main_dir_path, sub) for sub in os.listdir(val_main_dir_path) if
                           os.path.isdir(os.path.join(val_main_dir_path, sub))]
        all_subdirs.extend(deeper_sub_dirs)

        for deeper_sub_dir in deeper_sub_dirs:
            deeper_sub_dir_name = os.path.basename(deeper_sub_dir)
            missing_files = [file for file in required_files if
                             not os.path.exists(os.path.join(deeper_sub_dir, file))]

            if missing_files:
                raise ValueError(
                    f"Missing files in validation/{val_main_dir}/{deeper_sub_dir_name}: {missing_files}")

        # Check if all validations passed
    for main_dir, sub_validation in file_validation.items():
        for sub_dir, validation in sub_validation.items():
            if not validation["all_files_present"]:
                raise ValueError(f"Missing files in {main_dir}/{sub_dir}: {validation['missing_files']}")

    return existing_dirs, all_subdirs


def initialize_model(model_type):
    """
    Initialize and return the model based on the selected type.
    """
    if model_type == 'lgr':
        return LogisticRegression(max_iter=10000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, the supported types are: 'lgr'- you can add more models.")


def preprocess_all_projects(paths, output_pickle="projects_mimic_results.pkl"):
    project_results = {}
    data = pd.DataFrame()
    tag = pd.DataFrame()
    for path_to_read in paths:
        project_number = os.path.basename(path_to_read)
        tag_project = pd.read_csv(os.path.join(path_to_read, 'tag.csv'), index_col=0)
        tag_project = pd.Series(tag_project['Tag']).to_frame()
        tag_project.index = tag_project.index.astype(str)
        tag_project.index= tag_project.index+project_number

        for_preprocess = pd.read_csv(os.path.join(path_to_read, 'for_preprocess.csv'), index_col=0,
                                     low_memory=False)
        for_preprocess = for_preprocess.rename(columns={for_preprocess.columns[0]: 'ID'})
        for_preprocess.columns = for_preprocess.loc['taxonomy']
        for_preprocess = for_preprocess.drop(['taxonomy'])
        for_preprocess.index = for_preprocess.index.astype(str)
        for_preprocess.index = for_preprocess.index + project_number

        mutual = for_preprocess.index.intersection(tag_project.index)
        for_preprocess = for_preprocess.loc[mutual]
        tag_project = tag_project.loc[mutual]
        for_preprocess = for_preprocess.loc[mutual]
        for_preprocess = for_preprocess.loc[~for_preprocess.index.duplicated()]
        tag_project = tag_project.loc[mutual]
        tag_project = tag_project.loc[~tag_project.index.duplicated()]
        project_results[project_number] = {"tag": tag_project, "path_to_read": path_to_read}

        data = pd.concat([data, for_preprocess], axis=1)
        tag = pd.concat([tag, tag_project], axis=0)

    taxonomy_row = pd.Series(data.columns, index=data.columns, name='taxonomy')
    data = pd.concat([data, taxonomy_row.to_frame().T], axis=0)
    data = data.fillna(0)
    data = data.reset_index()

    data = data.rename(columns={data.columns[0]: 'ID'})
    processed, strain_flag = apply_mimic(
        "processed_all", folder='', tag=tag, mode="preprocess",
        preprocess=True, rawData=data, taxnomy_group='sub PCA', strain_flag=True
    )
    project_results["processed_all"] = {"processed": processed}
    pd.to_pickle(project_results, output_pickle)
    return processed, project_results
