import imp
import os 
from xml.etree.ElementInclude import include
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from src.utils.all_utils import read_yaml, create_directory
from tensorflow.keras.applications import MobileNetV2
import logging

def build_save_model(config_path: str, params_path: str):
    content = read_yaml(config_path=config_path)
    params = read_yaml(config_path=params_path)

    include_top = params["base"]["include_top"]
    weights = params["base"]["weights"]

    image_width, image_height, image_channels = params["base"]["image_width"], params["base"]["image_height"], params["base"]["image_channels"]

    if weights == "None":
        weights = None

    model = MobileNetV2(include_top=include_top,
                    weights=weights,
                    input_shape=(image_width,image_height,image_channels))
    
    logging.info("Successfully loaded the model")
    

    artifact_dir_name = content["artifacts"]["artifact_dir_name"]
    artifact_dir_path = os.path.join(os.getcwd(),artifact_dir_name)

    untrained_model_dir_name = content["artifacts"]["untrained_model_dir_name"]
    untrained_model_dir_path = os.path.join(artifact_dir_path,untrained_model_dir_name)

    create_directory([artifact_dir_path,untrained_model_dir_path])

    untrained_model_file_name = content["artifacts"]["untrained_model_file_name"]
    untrained_model_file_path = os.path.join(untrained_model_dir_path,untrained_model_file_name)

    model.save(untrained_model_file_path)
    logging.info(f"Successfully saved the model at {untrained_model_file_path}")
