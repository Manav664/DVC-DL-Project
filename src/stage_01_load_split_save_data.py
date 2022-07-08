from distutils.command.config import config
import imp
import numpy as np
import tensorflow as tf
import logging
import gc
import argparse

from src.utils.all_utils import *

def load_split_save(config_path: str):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    logging.info("Data loaded and splitted successfully")

    content = read_yaml(config_path=config_path)

    data_dir_name = content["data"]["data_dir_name"]
    data_dir_path = os.path.join(os.getcwd(),data_dir_name)
    create_directory([data_dir_name])

    train_images_file_name = content["data"]["train_images_file_name"]
    train_images_file_path = os.path.join(data_dir_path,train_images_file_name)
    test_images_file_name = content["data"]["test_images_file_name"]
    test_images_file_path = os.path.join(data_dir_path,test_images_file_name)
    train_labels_file_name = content["data"]["train_labels_file_name"]
    train_labels_file_path = os.path.join(data_dir_path,train_labels_file_name)
    test_labels_file_name = content["data"]["test_labels_file_name"]
    test_labels_file_path = os.path.join(data_dir_path,test_labels_file_name)
    
    np.save(train_images_file_path,x_train)
    np.save(test_images_file_path,x_test)
    np.save(train_labels_file_path,y_train)
    np.save(test_labels_file_path,y_test)
    logging.info("Data saved sucessfully")

    del x_train,y_train,x_test,y_test
    gc.collect()

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    
    default_config_path = os.path.join(os.path.join(os.getcwd(),"config"),"config.yaml")

    args.add_argument("--config","-c",default=default_config_path)

    parsed_args = args.parse_args()

    initialize_log(config_path=parsed_args.config,filemode="w")
    logging.info("\n <<<<<< stage 1 started")
    load_split_save(config_path=parsed_args.config)
    logging.info("\n stage 1 completed >>>>>>")




