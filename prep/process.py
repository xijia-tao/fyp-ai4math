import os
import json


files = []
dirs = [
    'data/MUSTARDSauce/data/subset_filtered',
    'data/MUSTARDSauce/data/subset_random'
]
filenames = [
    'filtered_the.json',
    'random_the.json'
]
for dir, filename in zip(dirs, filenames):
    for i, name in enumerate(os.listdir(dir)):
        path = f'{dir}/{name}'
        if os.path.isdir(path):
            continue
        file = json.load(open(f'{dir}/{name}'))
        
        informal_statement = file['informal_statement']
        formal_proof = file['formal_proof']
        informal_proof = file['informal_proof']

        type = name[:2]
        assert type in ['tp', 'wp']

        if type == 'tp':
            formal_statement = formal_proof[:formal_proof.rfind(':=')] + ':='
        else:
            formal_statement = None

        files.append(
            {
                'name': name,
                'type': type,
                'input': informal_statement,
                'intermediate': informal_proof,
                'output': formal_proof,
                'formal_statement': formal_statement
            }
        )
    json.dump(files, open('data/MUSTARDSauce/' + filename, 'w'))

