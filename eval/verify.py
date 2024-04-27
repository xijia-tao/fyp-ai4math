# interact with the Lean theorem prover to verify the proofs generated
import subprocess
import os
from tqdm import tqdm
from glob import glob


base_dir = 'minif2f-results/<checkpoint-name>' # results directory for a specific model under a experiment setting
ignores = ['history'] # ignore certain subdirectories that are for testing and backup

imports = """import minif2f_import

open_locale big_operators
open_locale nat
open_locale real
open_locale rat
"""

paths = [base_dir + '/' + filename for filename in os.listdir(base_dir) if filename.endswith('.lean')]

for path in tqdm(paths):
    content = open(path).read()
    txt_path = path[:-5] + '.txt'

    # skip sorry as the proof must be incorrect
    if 'sorry' in content:
        with open(txt_path, 'w') as f:
            f.write('Autoskip due to sorry\n')
        continue
    
    # stop proof verification when a 120-sec timeout is reached
    try:
        with open(txt_path, 'w') as f:
            subprocess.run(["lean", path], stdout=f, timeout=120) 
    except:
        with open(txt_path, 'w') as f:
            f.write('Autoskip due to timeout\n')
