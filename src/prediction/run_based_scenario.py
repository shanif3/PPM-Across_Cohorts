import os.path
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
import numpy as np
from .utils import *


def run_same_train_test(project_number, type, leave_one_out_flag, validation, project_mimic_result, substrings,
                        specific_bac,
                        confounders_names_list=None, model_type='lgr'):
    print(f"Running {project_number}\n")
    if substrings == None:
        # stop all the process and threads
        print("You must enter the substrings, try again.")
        sys.exit(1)

    path_to_read = project_mimic_result.get(project_number)['path_to_read']
    df = project_mimic_result.get(project_number)['processed']

    if confounders_names_list == None:
        tag = project_mimic_result.get(project_number)['tag']

    else:
        tag = pd.read_csv(os.path.join(path_to_read, 'full_tag.csv'), index_col=0)
        bio_to_add = tag[confounders_names_list]
        tag = pd.DataFrame(tag['Tag'], columns=['Tag'])

    if confounders_names_list != None:
        mutual = df.index.intersection(bio_to_add.index)
        bio_to_add = bio_to_add.loc[mutual]
        df = df.loc[mutual]

        df = pd.concat([df, bio_to_add], axis=1)

    tag = tag.dropna()
    tag.index = tag.index.astype(str)
    mutual = df.index.intersection(tag.index)
    df = df.loc[mutual]
    df = df.loc[~df.index.duplicated()]
    tag = tag.loc[mutual]
    tag = tag.loc[~tag.index.duplicated()]

    df_test, tag_test = None, None
    train_on = type

    results, relevant_bact = run_model(project_number, df, tag, df_test, tag_test, train_on, leave_one_out_flag,
                                         leave_one_dataset_out_flag=False,
                                         validation=validation, confounders_names_list=confounders_names_list,
                                         substrings=substrings,
                                         path_to_save=path_to_read, model_type=model_type, specific_bac=specific_bac)

    auc_folds, bac_coeff = results
    return auc_folds, bac_coeff, relevant_bact


def run_lodo(project_number, directories, specific_bac, train_on, validation, confounders,
             project_mimic_result, model_type, substrings, fair_pick):
    VALID_TRAIN_ON = ["16S", "Shotgun", "18S", "all"]
    if train_on not in VALID_TRAIN_ON:
        raise ValueError(f"Invalid train_on value: {train_on}. Must be one of {VALID_TRAIN_ON}")

    try:
        if train_on != 'all':
            projects_to_train_on = [sub for sub in os.listdir(directories[train_on]) if
                                    os.path.isdir(os.path.join(directories[train_on], sub))]
        else:
            projects_to_train_on = [sub for sub in os.listdir(directories['16S']) if
                                    os.path.isdir(os.path.join(directories['16S'], sub))] + [sub for sub in os.listdir(
                directories['Shotgun']) if os.path.isdir(os.path.join(directories['Shotgun'], sub))]

        projects_to_train_on = [project for project in projects_to_train_on if project != project_number]
        if 'PRJEB27564' in project_number:
            # drop the project contain PRJEB27564 in order to avoid leakage
            projects_to_train_on = [project for project in projects_to_train_on if 'PRJEB27564' not in project]

        if fair_pick:
            substrings = overlap_check(projects_to_train_on, project_mimic_result)
            if substrings == set():
                print("No overlap bact were found")

                return [0], '','No overlap bacteria'

        indexes=[]
        tag = pd.DataFrame()
        print(f"lodo is {project_number} and train on is {train_on}\n")
        for project in projects_to_train_on:
            tag_project = project_mimic_result.get(project)['tag']
            tag = pd.concat([tag, tag_project], axis=0)
            indexes.extend(tag_project.index.tolist())

        processed= project_mimic_result.get('processed_all')['processed']
        data = processed.loc[indexes]



        df_test = project_mimic_result.get(project_number)['processed']
        df_test_path_to_read = project_mimic_result.get(project_number)['path_to_read']
        tag_test = project_mimic_result.get(project_number)['tag']

        tag_test = tag_test.dropna()
        tag_test.index = tag_test.index.astype(str)

        project_number = re.sub(r'[\\/]', '_', project_number)

        (results), relevant_bact = run_model(project_number, data, tag, df_test, tag_test, train_on,
                                             leave_one_out_flag=False,
                                             leave_one_dataset_out_flag=True,
                                             validation=validation, confounders_names_list=confounders,
                                             path_to_save=df_test_path_to_read,
                                             substrings=substrings, model_type=model_type, specific_bac=specific_bac)

        auc_folds, bac_coeff = results
        print(f"Finished {project_number} with train on {train_on}\n")
        return auc_folds,bac_coeff, relevant_bact

    except Exception as e:
        print(f"Error processing run_lodo function, with project {project_number}: {e}")
        raise Exception(f"Error processing run_lodo function, with project {project_number}: {e}")


def main(project_number, directories, type, same_train_test, leave_one_dataset_out_flag, leave_one_out_flag,
         train_all, validation,
         confounders_names_list, project_mimic_result, model_type, substrings, specific_bac, fair_pick):
    if same_train_test:
        auc_folds, bac_coeff,substrings = run_same_train_test(project_number, type, leave_one_out_flag, validation,
                                                    project_mimic_result, substrings, specific_bac,
                                                    confounders_names_list,
                                                    model_type=model_type)

    elif leave_one_dataset_out_flag:
        auc_folds,bac_coeff, substrings = run_lodo(project_number, directories, specific_bac,
                                         train_on=train_all,
                                         validation=validation, confounders=confounders_names_list,
                                         project_mimic_result=project_mimic_result, model_type=model_type,
                                         substrings=substrings, fair_pick=fair_pick)

    return auc_folds, bac_coeff,substrings


def run_model(project_number, df, tag, df_test, tag_test, train_on, leave_one_out_flag,
              leave_one_dataset_out_flag, validation,
              path_to_save, substrings, confounders_names_list=None,
              model_type='lgr', specific_bac=False):
    if confounders_names_list != None:
        # add to substrings the confounders names
        substrings = substrings + confounders_names_list

    # if i want to take the specific bacteria
    if specific_bac:
        substrings = fix_substring(substrings)
        relevant_bac = [col for col in df.columns if
                        any(col.endswith(substring) for substring in substrings)]

    else:
        # if i want to take all the bacteria under this genus
        relevant_bac = [col for col in df.columns if any(substring in col for substring in substrings)]

    if relevant_bac:
        to_learn = df[relevant_bac]
        new_rel_bac = []
        for bac in relevant_bac:
            if confounders_names_list != None and bac in confounders_names_list:
                new_rel_bac.append(bac)
                continue

            if bac=='k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Betaproteobacteriales;f__Burkholderiaceae;g__Sutterella;Ambiguous_taxa_0;t__':
                bac='k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Betaproteobacteriales;f__Burkholderiaceae;g__Sutterella;s__Ambiguous_taxa_0;t__'
            if bac =='k__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Akkermansiaceae;g__Akkermansia;Ambiguous_taxa_0;t__':
                bac='k__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Akkermansiaceae;g__Akkermansia;s__Ambiguous_taxa_0;t__'
            bact_split = bac.split(';')[-3:]
            if len(bact_split) == 1:
                continue
            for index, bact_sp in enumerate(bact_split):
                try:
                    if bact_sp.startswith('g__'):
                        a= bact_split, bact_split
                        bact_split = bact_split[index:]
                        new_bac = bact_split
                        if 'Ambiguous' in new_bac[1]:
                            c=0
                        if len(bact_split) == 1:
                            new_rel_bac.append('g__' + bact_split[0].split('__')[1].split('_')[0])
                        else:
                            # first name of the specie means check if the specie has the genus name in the beginning and if so remove it
                            first_name_of_specie = bact_split[1].split('__')[1].split('_')[0]
                            if first_name_of_specie == 'Sutterella':
                                c=0
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
                except Exception as e:
                    new_rel_bac.append(';'.join(bact_split))
                    continue
                    print(f"Error processing run_model function, with project {project_number}: {e}")
                    raise Exception(f"Error processing run_model function, with project {project_number}: {e}")


                else:
                    continue
                break

        to_learn.columns = new_rel_bac
        to_learn = to_learn.groupby(to_learn.columns, axis=1).sum()

        return run_based_model(to_learn, tag, df_test, tag_test, project_number, train_on, leave_one_out_flag,
                               leave_one_dataset_out_flag, validation,
                               path_to_save, substrings, confounders_names_list, model_type, specific_bac), relevant_bac

    else:
        return 0,'', 'No relevant bacteria'


def run_based_model(to_learn, tag, df_test, tag_test, project_number, train_on, leave_one_out_flag,
                    leave_one_dataset_out_flag,
                    validation,
                    path_to_save, substrings, confounders_names_list, model_type, specific_bac):
    try:
        true_labels = []
        pred_probabilities = []
        auc_folds = []

        if leave_one_out_flag:
            loo = LeaveOneOut()
            y_pred_loo = np.zeros(len(tag))

            # Perform leave-one-out cross-validation
            for train_index, test_index in loo.split(to_learn.index):
                X_train, X_test = to_learn.iloc[train_index], to_learn.iloc[test_index]
                y_train, y_test = tag.iloc[train_index], tag.iloc[test_index]
                model = initialize_model(model_type)

                # Train the model on the training data

                model.fit(X_train, y_train)

                # Predict on the left-out test sample
                y_pred_loo[test_index] = model.predict_proba(X_test)[:, 1]

            pred_probabilities = y_pred_loo
            true_labels = tag['Tag'].values
            auc_folds.append(roc_auc_score(true_labels, pred_probabilities))

        elif not leave_one_out_flag and not leave_one_dataset_out_flag:
            try:
                skf = StratifiedKFold(n_splits=10)
                for train_index, test_index in skf.split(to_learn, tag):
                    X_train, y_train = to_learn.iloc[train_index], tag.iloc[train_index]
                    X_test, y_test = to_learn.iloc[test_index], tag.iloc[test_index]
                    model = initialize_model(model_type)

                    # Train model on full training set after LeaveOneOut fitting
                    model.fit(X_train, y_train)
                    true_labels.extend(y_test.values)
                    y_scores = model.predict_proba(X_test)[:, 1]
                    pred_probabilities.extend(y_scores)
                    auc_folds.append(roc_auc_score(y_test, y_scores))
            except Exception as e:
                print(
                    f"Cannot run stratified kfold for {project_number}, there is not many samples to split, set skf auc score to 0")
                return [0],''


        elif leave_one_dataset_out_flag:
            if validation:
                c=0
            model = initialize_model(model_type)
            if model_type == 'lgbm' or model_type == 'xgb':
                to_learn=to_learn.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

            model.fit(to_learn, tag)

            if specific_bac:
                # substrings = fix_substring(substrings, strain_flag)
                relevant_bac_test = [col for col in df_test.columns if
                                     any(col.endswith(substring) for substring in substrings)]
            else:
                # # if i want to take all the bacteria under this genus
                relevant_bac_test = [col for col in df_test.columns if
                                     any(substring in col for substring in substrings)]

            df_test = df_test[relevant_bac_test]

            new_rel_bac = fix_and_cut_in_genara(relevant_bac_test)

            df_test.columns = new_rel_bac
            df_test = df_test.groupby(df_test.columns, axis=1).sum()

            common_relevant_bac = list(to_learn.columns.intersection(df_test.columns))
            df_test = df_test[common_relevant_bac]
            diff = list(set(to_learn.columns) - set(df_test.columns))
            if diff:
                # create col of the diff columns and fill with -1
                for col in diff:
                    df_test[col] = -1

            # make sure the columns are in the same order
            df_test = df_test[to_learn.columns]
            true_labels = tag_test
            y_test = model.predict_proba(df_test)[:, 1]
            pred_probabilities.extend(y_test)
            auc_folds.append(roc_auc_score(true_labels, y_test))

        extend_name, loo_flag_name = get_name_to_use(leave_one_out_flag, leave_one_dataset_out_flag,
                                                     confounders_names_list, project_number, train_on, validation)

        if model_type == 'lgr':
            mean_coefficients = model.coef_[0]
            feature_names = to_learn.columns.tolist()
            bac_coeff = ", ".join(f"{bacteria}:{coeff:.4f}" for bacteria, coeff in zip(feature_names, mean_coefficients))


            save_coefficients_to_csv(mean_coefficients, feature_names,
                                     os.path.join(path_to_save, f'{extend_name}coefficients{loo_flag_name}.csv'))
        plot_roc(path_to_save, train_on, pred_probabilities, true_labels, loo_flag_name, leave_one_dataset_out_flag,
                 project_number, validation, confounders_names_list)
    except Exception as e:
        print(f"Error processing run_based_model function, with project {project_number}: {e}")
        return 'E',''

    if model_type == 'lgr':
        return auc_folds, bac_coeff
    return auc_folds
