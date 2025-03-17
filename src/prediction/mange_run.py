import os.path
from .run_based_scenario import main
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from .utils import *

global auc_results, lock
from src.miMic.mimic_test import apply_mimic

# Initialize a thread lock
lock = Lock()  # Lock for thread-safe updates


# Function to process datasets
def process_project(project_path, project_type, directories, relevant_bac, simulation_name, specific_bac, fair_pick,
                    scenario_name,
                    same_train_test, llo, lodo, train_all
                    , project_mimic_result, model_type):
    # Call the main function
    if project_type == 'validation':
        validation = True
    else:
        validation = False
    project_number = os.path.basename(project_path)
    if project_number=='OK141_schizo':
        c=0

    auc_folds, bac_coeff, substrings = main(project_number, directories, project_type, same_train_test, lodo,
                                 llo, train_all, validation=validation, confounders_names_list=None,
                                 project_mimic_result=project_mimic_result, model_type=model_type,
                                 substrings=relevant_bac, specific_bac=specific_bac, fair_pick=fair_pick)

    substrings = ';'.join(substrings)
    if auc_folds is not None:
        with lock:  # Ensure thread-safe updates
            for fold_idx, auc in enumerate(auc_folds):
                auc_results["Project"].append(project_number)
                auc_results["Type"].append(project_type)
                auc_results["Scenario"].append(scenario_name)
                auc_results["Fold"].append(fold_idx + 1)
                auc_results["AUC"].append(auc)
                auc_results["Substrings"].append(substrings)
                auc_results['Coefficients'].append(bac_coeff)

            # Save intermediate results to CSV
            pd.DataFrame(auc_results).to_csv(
                f"{simulation_name}.csv", index=False)


# Define scenarios for each project with parallel processing
def process_all_scenarios_parallel(project_path, project_mimic_result, directories, model_type, relevant_bac,
                                   simulation_name, specific_bac, fair_pick):
    project_type = os.path.split(os.path.split(project_path)[0])[1]

    train_all_value = '16S' if project_type == '16S' else 'Shotgun'

    scenarios = [
        {"scenario_name": "10K folds", "same_train_test": True, "llo": False, "lodo": False, "train_all": "None",
         'project_mimic_result': project_mimic_result, 'model_type': model_type},
        {"scenario_name": "Leave one sample out", "same_train_test": True, "llo": True, "lodo": False,
         "train_all": "None", 'project_mimic_result': project_mimic_result, 'model_type': model_type},
        {"scenario_name": "Leave one dataset out; train on-Shotgun+16S", "same_train_test": False, "llo": False,
         "lodo": True, "train_all": 'all', 'project_mimic_result': project_mimic_result,
         'model_type': model_type},
        {"scenario_name": f"Leave one dataset out; train on-{train_all_value}", "same_train_test": False, "llo": False,
         "lodo": True, "train_all": train_all_value, 'project_mimic_result': project_mimic_result,
         'model_type': model_type}
    ]

    with ThreadPoolExecutor() as executor:
        executor.map(
            lambda args: process_project(project_path, project_type, directories, relevant_bac, simulation_name,
                                         specific_bac, fair_pick, **args), scenarios)


def prepare_mimic_parallel(project_number, processed_project, tag_project, path_to_read):
    folder = os.path.join(path_to_read, 'images')

    try:
        _, _, df_corrs = apply_mimic(
            project_number, folder, tag_project, eval="man", threshold_p=0.05,
            processed=processed_project, apply_samba=True, save=True, strain_flag=True)

    except Exception as e:
        print(f"Error processing MIMIC {project_number}: {e}")
        df_corrs = None

    return project_number, {"df_corrs": df_corrs, "processed": processed_project, "tag": tag_project}


def apply_mimic_parallel(paths, output_pickle="projects_mimic_results.pkl"):
    processed,project_results = preprocess_all_projects(paths, output_pickle=output_pickle)

    def update_results(future):
        project_number, result = future.result()
        with lock:  # Ensure thread-safe updates
            project_results[project_number].update(result)

    with ThreadPoolExecutor() as executor:
        futures = []

        # Process shotgun datasets
        for path_to_read in paths:
            project_number = os.path.basename(path_to_read)
            tag = project_results[project_number]["tag"]
            mutual = processed.index.intersection(tag.index)
            processed_project = processed.loc[mutual]
            tag_project = tag.loc[mutual]
            # Drop columns where that are
            processed_project = processed_project.loc[:, processed_project.nunique() > 1]
            # drop duplicates
            processed_project = processed_project.loc[~processed_project.index.duplicated(keep='first')]
            tag_project = tag_project.loc[~tag_project.index.duplicated(keep='first')]

            print(f"Processing {project_number}...")
            future = executor.submit(
                prepare_mimic_parallel, project_number, processed_project, tag_project, path_to_read)
            futures.append(future)

        # Wait for all futures and update results
        for future in futures:
            future.add_done_callback(update_results)

    # Save results to a pickle file
    pd.to_pickle(project_results, output_pickle)

    return project_results


# Main function
def run(relevant_bac, simulation_name, model_type, specific_bac, fair_pick,path_to_read):
    # Dictionary to store AUC fold results
    global auc_results
    auc_results = {"Project": [], "Type": [], "Scenario": [], "Fold": [], "AUC": [], "Substrings": [], 'Coefficients': []}

    directories, projects_path = check_required_directories(path_to_read)

    # preform mimic for each dataset
    project_mimic_result = apply_mimic_parallel(projects_path, output_pickle="projects_mimic_results.pkl")

    # Process all datasets
    for project_number in projects_path:
        process_all_scenarios_parallel(project_number, project_mimic_result, directories, model_type=model_type,
                                           relevant_bac=relevant_bac, simulation_name=simulation_name,
                                           specific_bac=specific_bac, fair_pick=fair_pick)


