import os
import pandas as pd


def create_directory_tree_dict(directory_path):
    directory_tree = {}
    for root, directories, files in os.walk(directory_path):
        current_dict = directory_tree
        for directory in root.split('/'):
            current_dict = current_dict.setdefault(directory, {})
        for file in files:
            current_dict[file] = None
    return directory_tree


def flatten_directory_tree(directory_tree, parent_path='', sep='/'):
    flattened = []
    for item, value in directory_tree.items():
        current_path = os.path.join(parent_path, item)
        if isinstance(value, dict):
            flattened.extend(flatten_directory_tree(value, current_path, sep=sep))
        else:
            flattened.append(current_path)
    return flattened


directory_path = '/Users/hoanglinh96nl/Documents/conf/appearance'
directory_tree_dict = create_directory_tree_dict(directory_path)
flattened_data = flatten_directory_tree(directory_tree_dict)

df = pd.DataFrame(flattened_data, columns=['Path'])
df = df['Path'].str.split('/', expand=True)
df.to_excel('all_dirs.xlsx')
