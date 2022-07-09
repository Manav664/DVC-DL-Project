import argparse
import imp
import os
from src.utils.all_utils import *
from src.utils.callbacks import * 

if __name__ == '__main__' :



    args = argparse.ArgumentParser()
    
    default_config_path = os.path.join(os.path.join(os.getcwd(),"config"),"config.yaml")

    args.add_argument("--config","-c",default=default_config_path)

    parsed_args = args.parse_args()

    initialize_log(config_path=parsed_args.config,filemode="a")

    logging.info("\n <<<<<< stage 3 started ")
    try:
        content = read_yaml(parsed_args.config)
        callback_dir_name = content["artifacts"]["callback_dir_name"]
        tensorboard_dir_name =  content["artifacts"]["tensorboard_dir_name"]
        checkpoint_dir_name = content["artifacts"]["checkpoint_dir_name"]

        artifact_dir_name = content["artifacts"]["artifact_dir_name"]
        artifact_dir_path = os.path.join(os.getcwd(),artifact_dir_name)

        callback_dir_path = os.path.join(artifact_dir_path,callback_dir_name)
        tensorboard_dir_path = os.path.join(artifact_dir_path,tensorboard_dir_name)
        checkpoint_dir_path = os.path.join(artifact_dir_path,checkpoint_dir_name)

        create_directory([callback_dir_path,tensorboard_dir_path,checkpoint_dir_path])

        create_save_tensorboard_cb(tensorboard_dir_path,callback_dir_path)
        create_save_model_ckp(checkpoint_dir_path,callback_dir_path)

        logging.info("\n stage 3 completed >>>>>>")
    except Exception as e:
        logging.info(f"FAILED to create callbacks")
        logging.info(f"{e}")