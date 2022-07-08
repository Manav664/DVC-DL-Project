
import yaml
import os
import logging

def read_yaml(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def initialize_log(config_path: str, filemode="a"):
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

    content = read_yaml(config_path=config_path)
    log_dir_name = content["logs"]["log_dir_name"]
    log_dir_path = os.path.join(os.getcwd(),log_dir_name)
    log_file_name = content["logs"]["log_file_name"]
    log_file_path = os.path.join(log_dir_path,log_file_name)
    create_directory([log_dir_path])

    logging.basicConfig(filename=log_file_path,
                        filemode=filemode,
                        format=logging_str,
                        level=logging.INFO)
    

def create_directory(dir_paths: list):

    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
