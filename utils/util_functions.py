import os
import json


def append_folder_name_to_path(path, folder_name):
    return os.path.join(path, folder_name)

def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok = True)

def save_dict_to_json_file(dict, file_name, folder_path):
    file_path = append_folder_name_to_path(folder_path, file_name + '.json')
    with open(file_path, 'w') as file:
        json.dump(dict, file, indent = 4, separators = (', ', ': '))

def save_json_string_to_file(json_string, file_name, folder_path):
    file_path = append_folder_name_to_path(folder_path, file_name + '.json')
    with open(file_path, 'w') as file:
        file.write(json_string)

def load_json_file_as_dict(folder_path, file_name):
    file_path = append_folder_name_to_path(folder_path, file_name + '.json')
    with open(file_path, 'r') as file:
        dict = json.load(file)
    return dict

def load_json_file_as_string(folder_path, file_name):
    loaded_dict = load_json_file_as_dict(folder_path, file_name)
    return json.loads(loaded_dict)