# Cross-cohort microbiome based prediction of Parkinson Disease using six consistent genera
This repository implements of the paper ["Cross-cohort microbiome-based prediction pipeline for Parkinson's Disease (PD) using six consistent genera"]().
The goal of this work is to predict PD based on microbiome data across different cohorts using machine learning models.
## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#Project-structure)

## Overview

This repository provides functionality for running a machine learning model that performs predictions using genus-specific microbiome data. It supports various configurations like:
- Choosing between different **model types** (`lgr` as an example).
- Selecting whether to pick **specific genera** or consider a broader set of genera.
- Configuring **fair training selection** or selecting specific genera for training based on a predefined list.

## Installation

To use the pipeline, clone the repository and install the necessary dependencies. This can be done by following these steps:

### Clone the repository
```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

## Usage
The repository provides a `main.py` script that can be used to run the pipeline. The script requires the following parameters:
- `model_type`: The type of model to use for training. For example, `lgr`.
- `specific_bac`: Whether to use specific genera for training or to use the species and strains under those genera. 
- `fair_pick`: Whether to use fair genera selection while training or not and use the user specified genera.
- `relevant_bac`: (optional): A list of genera to use for training. This is only required if `specific_bac` is set to `True`.
- `path_to_save`: path to save the auc score for each dataset and each scenario.

## Project Structure
The repository has the following folder structure:

```
project_root/
├── Data/
│   ├── 16S/
│   ├── Shotgun/
│   ├── validation/
│   │   ├── 16S/
│   │   ├── Shotgun/
├── src/
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── mange_run.py
│   │   ├── run_based_scenario.py
│   │   ├── utils.py
│   ├── MIPMLP_package/
│   ├── plotting/
│   ├── miMic.py
├── main.py
├── README.md
├── requirements.txt
```

### Data Directory Structure
In the `Data/` directory, there are three main subdirectories:

1. `16S/`: Contains datasets related to 16S sequencing.
2. `Shotgun/`: Contains datasets related to Shotgun sequencing.
3. `validation/`: Contains datasets to be used for validation. This is split into two subdirectories:
   1. `16S/`: Contains validation datasets for 16S sequencing.
   2. `Shotgun/`: Contains validation datasets for Shotgun sequencing.

### Organizing Your Datasets
Each dataset must be organized into its own directory within either `16S/`, `Shotgun/`, or `validation/`. Each dataset directory should contain exactly two files:

1. `for_preprocess.csv`: This file contains the raw ASVs table where the rows represent individual samples, and the columns represent the different microbes. The last row should be labeled `taxonomy` and contain the names of the microbes. (For more details see [MILMLP](https://pypi.org/project/MIPMLP/), you can see examples in `Data/` directory).

2. `tag.csv`: This file contains the target labels for the corresponding samples. It should have a column named `Tag` that contains the target labels.