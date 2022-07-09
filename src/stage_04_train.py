import argparse
from gc import callbacks
import imp
import os
import numpy as np
import tensorflow as tf
from src.utils.all_utils import *
from src.utils.callbacks import * 

def train_model(config_path:str, params_path:str):
    content = read_yaml(config_path=config_path)
    params = read_yaml(config_path=params_path)

    artifact_dir_name = content["artifacts"]["artifact_dir_name"]
    artifact_dir_path = os.path.join(os.getcwd(),artifact_dir_name)

    
    #loading data
    logging.info("Fetching data")
    data_dir_name = content["data"]["data_dir_name"]
    data_dir_path = os.path.join(os.getcwd(),data_dir_name)
    train_images_file_name = content["data"]["train_images_file_name"]
    train_images_file_path = os.path.join(data_dir_path,train_images_file_name)
    test_images_file_name = content["data"]["test_images_file_name"]
    test_images_file_path = os.path.join(data_dir_path,test_images_file_name)
    train_labels_file_name = content["data"]["train_labels_file_name"]
    train_labels_file_path = os.path.join(data_dir_path,train_labels_file_name)
    test_labels_file_name = content["data"]["test_labels_file_name"]
    test_labels_file_path = os.path.join(data_dir_path,test_labels_file_name)

    x_train = np.load(train_images_file_path)
    y_train = np.load(train_labels_file_path)
    x_test = np.load(test_images_file_path)
    y_test = np.load(test_labels_file_path)
    logging.info("Data Fetched Sucessfully")

    #getting callbacks 
    callback_dir_name = content["artifacts"]["callback_dir_name"]
    callback_dir_path = os.path.join(artifact_dir_path,callback_dir_name)
    callbacks = prepare_callbacks(callback_dir_path)

    #loading the model
    logging.info("Loading the model")
    untrained_model_dir_name = content["artifacts"]["untrained_model_dir_name"]
    untrained_model_dir_path = os.path.join(artifact_dir_path,untrained_model_dir_name)
    untrained_model_file_name = content["artifacts"]["untrained_model_file_name"]
    untrained_model_file_path = os.path.join(untrained_model_dir_path,untrained_model_file_name)

    model = tf.keras.models.load_model(untrained_model_file_path)
    logging.info("Model loaded")

    #getting training parameters
    epochs = params["train_model"]["epochs"]
    batch_size = params["train_model"]["batch_size"]

    #training the model
    logging.info("Model training started")
    model.fit(x_train,y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data= (x_test,y_test))
    
    logging.info("Model training completed")

    #saving the trained model
    trained_model_dir_name = content["artifacts"]["trained_model_dir_name"]
    trained_model_dir_path = os.path.join(artifact_dir_path,trained_model_dir_name)
    trained_model_file_name = content["artifacts"]["trained_model_file_name"]
    trained_model_file_path = os.path.join(trained_model_dir_path,trained_model_file_name)
    
    logging.info("Saving the trained model")
    model.save(trained_model_file_path)
    logging.info("Model saved successfully")
    

  

if __name__ == '__main__' :



    args = argparse.ArgumentParser()
    
    default_config_path = os.path.join(os.path.join(os.getcwd(),"config"),"config.yaml")

    defalt_param_path = os.path.join(os.getcwd(),"params.yaml")

    args.add_argument("--config","-c",default=default_config_path)

    args.add_argument("--params","-p",default=defalt_param_path)

    parsed_args = args.parse_args()

    initialize_log(config_path=parsed_args.config,filemode="a")

    logging.info("\n <<<<<< stage 4 started ")
    try:
        train_model(parsed_args.config,parsed_args.params)
        logging.info("\n stage 4 completed >>>>>>")
    except Exception as e:
        logging.info(f"FAILED to train model")
        logging.info(f"{e}")