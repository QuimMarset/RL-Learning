import json
import os

def save_dict_to_json(dict, dict_name, path):
    path = os.path.join(path, dict_name + '.json')
    with open(path, 'w') as file:
        json.dump(dict, file)

def load_json_as_dict(path, dict_name):
    dict = None
    path = os.path.join(path, dict_name)
    with open(path, 'r') as file:
        dict = json.load(file)
    return dict