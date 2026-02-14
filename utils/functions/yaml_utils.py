import ruamel.yaml
import os

def initialize_yaml_parser():
    parser = ruamel.yaml.YAML()
    parser.default_flow_style = False
    parser.boolean_representation = ['False', 'True']
    return parser

def load_yaml_config(yaml_parser, file_path):
    with open(file_path, 'r') as stream:
        config_dict = yaml_parser.load(stream)
    return config_dict

def save_yaml_config(yaml_parser, file_path, config_dict):  
    with open(file_path, 'w') as stream:
        yaml_parser.dump(config_dict, stream)

def convert_list(list):
    s = ruamel.yaml.comments.CommentedSeq(list)
    s.fa.set_flow_style()
    return s

def insert_in_all_configs(*section_keys, value=None, dir_path="Configs/Training"):
    '''Updates all yaml files in a directory with the given value on the section specified by the section_keys argument.'''

    if not os.path.exists(dir_path):
        raise Exception("Invalid Path")
    
    parser = initialize_yaml_parser()

    yaml_extensions = (".yml", ".yaml")
    for filename in os.listdir(dir_path):
        config_path = os.path.join(dir_path, filename)
        extension = os.path.splitext(config_path)[-1].lower()

        if extension in yaml_extensions:
            config_dict = load_yaml_config(parser, config_path)
            cur_section = config_dict
            for key in section_keys[:-1]:
                if key not in cur_section:
                    cur_section[key] = {} 
                cur_section = cur_section[key]
            
            if isinstance(value, list):
                value = convert_list(value)

            cur_section[section_keys[-1]] = value

            save_yaml_config(parser, config_path, config_dict)
    return

def remove_from_all_configs(*section_keys, dir_path="Configs/Training"):
    '''Removes entry with the specified section keys, from all the configs in a directory'''
    
    if not os.path.exists(dir_path):
        raise Exception("Invalid Path")
    
    parser = initialize_yaml_parser()

    yaml_extensions = (".yml", ".yaml")
    for filename in os.listdir(dir_path):
        config_path = os.path.join(dir_path, filename)
        extension = os.path.splitext(config_path)[-1].lower()

        if extension in yaml_extensions:
            config_dict = load_yaml_config(parser, config_path)
            cur_section = config_dict
            for key in section_keys[:-1]:
                cur_section = cur_section[key]
            
            del cur_section[section_keys[-1]]

            save_yaml_config(parser, config_path, config_dict)
    return
