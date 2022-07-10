import imp
from multiprocessing.dummy import active_children
from operator import mod
import os 
from xml.etree.ElementInclude import include

from sklearn import metrics
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from src.utils.all_utils import read_yaml, create_directory, save_model_summary
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
import logging



def build_save_model(config_path: str, params_path: str):
    content = read_yaml(config_path=config_path)
    params = read_yaml(config_path=params_path)

    include_top = params["base"]["include_top"]
    weights = params["base"]["weights"]

    image_width, image_height, image_channels = params["base"]["image_width"], params["base"]["image_height"], params["base"]["image_channels"]

    if weights == "None":
        weights = None

    base_model = MobileNetV2(include_top=include_top,
                    weights=weights,
                    input_shape=(image_width,image_height,image_channels))

    num_classes = params["base"]["num_classes"]
    
    model = Sequential()

    model.add(Input((32,32,3)))
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes,activation='softmax'))

    logging.info("Successfully loaded the model")

    model_summary_str = save_model_summary(model=model)
    logging.info(f"Full model summary : \n {model_summary_str}") 

    optimizer = params["base"]["optimizer"]
    loss_fnc = params["base"]["loss_fnc"]

    model.compile(optimizer=optimizer,loss=loss_fnc,metrics=["accuracy"])

    artifact_dir_name = content["artifacts"]["artifact_dir_name"]
    artifact_dir_path = os.path.join(os.getcwd(),artifact_dir_name)

    untrained_model_dir_name = content["artifacts"]["untrained_model_dir_name"]
    untrained_model_dir_path = os.path.join(artifact_dir_path,untrained_model_dir_name)

    create_directory([artifact_dir_path,untrained_model_dir_path])

    untrained_model_file_name = content["artifacts"]["untrained_model_file_name"]
    untrained_model_file_path = os.path.join(untrained_model_dir_path,untrained_model_file_name)

    model.save(untrained_model_file_path) 
    logging.info(f"Successfully saved the model at {untrained_model_file_path}")
