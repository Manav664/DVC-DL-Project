import imp
from tabnanny import check
import time
from turtle import RawTurtle
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import joblib
import logging

def get_time_stamp(name: str) -> str:
    time_stamp  = time.asctime().replace(" ","_").replace("__","_").replace(":","_")
    time_stamp = name + "_" + time_stamp
    return time_stamp


def create_save_tensorboard_cb(tensorboard_dir: str, callback_dir: str):

    logging.info("creating tensorboard callback ...")

    unique_name = get_time_stamp("tb_logs")

    tb_running_dir = os.path.join(tensorboard_dir,unique_name)
    tensorboard_cb = TensorBoard(log_dir=tb_running_dir)
    
    joblib.dump(tensorboard_cb,
                os.path.join(callback_dir,"tensorboard.cb"))
    
    logging.info("Tensorboard callback created and saved")


def create_save_model_ckp(checkpoint_dir: str,callback_dir: str):

    logging.info("creating ModelCheckpoint callback ...")

    ckp_filepath = os.path.join(checkpoint_dir,"ckpt.h5")

    checkpoint_cb = ModelCheckpoint(filepath=ckp_filepath,
                                    save_best_only = True)
    
    joblib.dump(checkpoint_cb,os.path.join(callback_dir,"checkpoint.cb"))

    logging.info("ModelCheckpoint callback created and saved")

def prepare_callbacks(callback_dir):
    
    logging.info("Loading callbacks ..")
    callbacks_path = [os.path.join(callback_dir,cur_cb) for cur_cb in os.listdir(callback_dir) if cur_cb.endswith(".cb")]
    callbacks = [joblib.load(cur_cb) for cur_cb in callbacks_path]
    logging.info("callbacks loaded")
    return callbacks