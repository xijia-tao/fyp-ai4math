# after obtaining model responses, extract proofs from the responses into .lean files
import json
import re
import os


imports = """import minif2f_import

open_locale big_operators
open_locale nat
open_locale real
open_locale rat

"""

base_dir = 'minif2f-results'
TEST_FILE = 'minif2f/lean/src/test.lean'

# extract minif2f test theorems
test_dataset = open(TEST_FILE)
pattern = r'theorem((?:.|\n)*?):='
theorems = [x.group() for x in re.finditer(pattern, test_dataset.read())]

# multiple candidate patterns for lean code
pattern = r'```lean((?:.|\n)*?)```'
pattern2 = r'```Lean((?:.|\n)*?)```'
pattern3 = r'```\ntheorem((?:.|\n)*?)```'
beginend = r'begin\n((?:.|\n)*?)\nend'

def extract_proof(text, i=None, filename=None, theorem=None, use_first=True):
    assert i is not None or theorem_name is not None
    if i is not None:
        theorem = theorems[i]
    
    failed = 0
    codes = [x.group(1).strip() for x in re.finditer(pattern, text)]
    codes2 = [x.group(1).strip() for x in re.finditer(pattern2, text)]
    codes3 = [x.group(1).strip() for x in re.finditer(pattern3, text)]
    if len(codes) + len(codes2) + len(codes3) == 0:
        print('Warning: no code found')
        if 'gpt35' in filename:
            # heuristic due to the special response format of gpt-3.5
            code = text
            if (('theorem' not in c) and ('lemma' not in c) and ('example' not in c)) \
                or (theorem not in c):
                code = theorems[i] + '\nbegin\nsorry\nend'
                failed = 1
        else:
            code = theorems[i] + '\nbegin\nsorry\nend'
            failed = 1
    else:
        # extracts the first or last proof found
        if use_first:
            all_codes = codes + codes2 + codes3
        else:
            all_codes = codes[::-1] + codes2[::-1] + codes3[::-1]
        for c in all_codes:
            if (('theorem' not in c) and ('lemma' not in c) and ('example' not in c)) \
                or (theorem not in c):
                continue
            else:
                code = c
                break
        else:
            # extract begin...end from code
            codes = [x.group(1).strip() for x in re.finditer(beginend, text)]
            if len(codes) > 0:
                code = theorems[i] + '\nbegin\n' + codes[0] + '\nend'
                print(code, theorems[i])
            else:
                code = theorems[i] + '\nbegin\nsorry\nend'
                failed = 1
    
    # import necessary lean libraries
    content = code if 'import' in code else imports + code
    # remove #eval lines
    if "#eval" in content:
        content = '\n'.join([line for line in content.split('\n') if not line.startswith('#eval')])
    
    return content, failed

# begin proof extraction by separately considering for corex and other methods
if 'ma' in base_dir: # corex
    # flatten batched result
    for filename in os.listdir(base_dir):
        if not filename.endswith('.json') or 'all_results' in filename:
            continue
        file = json.load(open(f'{base_dir}/{filename}'))
        new_file = []

        # if already flattened 
        if type(file[0]['r1']['Tom']) == str:
            break
    
        for d in file:
            for i in range(len(d['r1']['Tom'])):
                new_file.append(
                    {
                        'r1': {'Tom': d['r1']['Tom'][i], 'Jerry': d['r1']['Jerry'][i]},
                        'r2': {'Tom': d['r2']['Tom'][i], 'Jerry': d['r2']['Jerry'][i]}
                    }
                )
        json.dump(new_file, open(f'{base_dir}/{filename}', 'w'))


    for filename in os.listdir(base_dir):
        if not filename.endswith('.json') or 'all_results' in filename \
            or os.path.exists(f'{base_dir}/{filename[:-5]}'):
            continue
    
        file = json.load(open(f'{base_dir}/{filename}'))
        
        if not os.path.exists(f'{base_dir}/{filename[:-5]}'):
            os.makedirs(f'{base_dir}/{filename[:-5]}')

        for i, d in enumerate(file):            
            r2_tom = d['r2']['Tom']
            r2_jerry = d['r2']['Jerry']
            
            code, _ = extract_proof(r2_tom, i, filename)
            with open(f'{base_dir}/{filename[:-5]}/{theorems[i].split()[1]}_t2.lean', 'w') as f:
                content = code if 'import' in code else imports + code
           
            code, _ = extract_proof(r2_jerry, i, filename)
            with open(f'{base_dir}/{filename[:-5]}/{theorems[i].split()[1]}_j2.lean', 'w') as f:
                f.write(code + '\n')
else: # other methods
    for filename in os.listdir(base_dir):
        if not filename.endswith('.json') or os.path.exists(f'{base_dir}/{filename[:-5]}')\
            or 'all_results' in filename:
            continue
        
        file = json.load(open(f'{base_dir}/{filename}'))
        if len(file) != 244:
            print("Warning: not all theorems are present in", filename)
            print("Skipping", filename)
            continue

        if not os.path.exists(f'{base_dir}/{filename[:-5]}'):
            os.makedirs(f'{base_dir}/{filename[:-5]}')

        failed = 0
        for i, d in enumerate(file):
            if 'rag' in base_dir and type(d) == dict:
                    d = d['answer']
            code, f = extract_proof(d, i, filename, use_first=False)
            failed += f

            theorem_name = theorems[i].split()[1]
            with open(f'{base_dir}/{filename[:-5]}/{theorem_name}.lean', 'w') as f:
                f.write(code + '\n')
        print(filename, "Number of theorems without extracted proofs:", failed)


