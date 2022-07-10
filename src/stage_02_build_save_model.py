import yaml
import argparse
from src.utils.all_utils import *
from src.utils.models import build_save_model

if __name__ == '__main__' :



    args = argparse.ArgumentParser()
    
    default_config_path = os.path.join(os.path.join(os.getcwd(),"config"),"config.yaml")

    defalt_param_path = os.path.join(os.getcwd(),"params.yaml")

    args.add_argument("--config","-c",default=default_config_path)

    args.add_argument("--params","-p",default=defalt_param_path)

    parsed_args = args.parse_args()

    initialize_log(config_path=parsed_args.config,filemode="a")

    logging.info("\n <<<<<< stage 2 started ")
    try:
        build_save_model(config_path=parsed_args.config,params_path=parsed_args.params)
        logging.info("\n stage 2 completed >>>>>>")
    except Exception as e:
        logging.info(f"FAILED to build and save model")
        logging.info(f"{e}")
