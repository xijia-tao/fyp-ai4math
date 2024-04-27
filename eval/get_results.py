# generate pass/fail results for the minif2f test set given a results directory
import json
import os


base_dir = 'minif2f-results' # results directory
ignores = ['history'] # ignore certain subdirectories that are for testing and backup

paths = []

all_results = {}
for subdir in os.listdir(base_dir):
    # continue if subdir is not a dir
    if not os.path.isdir(os.path.join(base_dir, subdir)) or subdir in ignores:
        continue

    print("Examining", subdir)
    success = []
    results = {'success': 0, 'timeout': 0, 'sorry': 0, 'failed': 0, 'total_failed': 0}

    for filename in os.listdir(os.path.join(base_dir, subdir)):
        if filename.endswith('.txt'):
            file = open(os.path.join(base_dir, subdir, filename)).read()
            # classify different sources of errors
            if 'Autoskip due to sorry' in file:
                results['sorry'] += 1
            elif 'Autoskip due to timeout' in file:
                results['timeout'] += 1
            elif len(file.strip()) == 0:
                results['success'] += 1
                success.append(os.path.join(base_dir, subdir, filename))
            else:
                results['failed'] += 1
            results['total_failed'] = results['failed'] + results['sorry'] + results['timeout']
    results['success_files'] = success
    print(results)
    all_results[subdir] = results

json.dump(all_results, open(base_dir + '/all_results.json', 'w'))
