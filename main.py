from src.prediction.mange_run import run
import warnings
import os
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
model_type= 'lgr'
specific_bac= False
fair_pick= True
path_to_read= r'C:\Users\user\PycharmProjects\pythonProject2\PPM-Across_Cohorts\Data'
relevant_bac= [
        "g__Roseburia",
        "g__Bifidobacterium",
        "g__Blautia",
        "g__Lactobacillus",
        "g__Akkermansia",
        "g__Faecalibacterium",
    ]

if fair_pick:
        sancerio= 'fair_selecting_based_training'
else:
        sancerio= 'picking_specific_bac_'

if specific_bac:
        spc_bac='specific_bac'
else:
        spc_bac='under_bac'
relevant_bac_name= '_'.join(relevant_bac)

path= os.path.join('save_results','genera',f'{model_type}',f'{sancerio}',f'{spc_bac}')
if not os.path.exists(path):
    os.makedirs(path)

simulation_name= os.path.join(f'{path}',f'{relevant_bac_name}')

run(relevant_bac,simulation_name=simulation_name, model_type=model_type, specific_bac=specific_bac,fair_pick=fair_pick, path_to_read=path_to_read)
