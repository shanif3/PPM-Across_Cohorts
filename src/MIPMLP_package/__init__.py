import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_src = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root_src)
from MIPMLP_package.service import preprocess
