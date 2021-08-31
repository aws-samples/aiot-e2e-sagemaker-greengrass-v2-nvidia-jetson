import os
import dlr
import cv2
import numpy as np
from dlr import DLRModel

import argparse
import time
import glob
from datetime import datetime
import config_utils
import IPCUtils as ipc_utils
import awsiot.greengrasscoreipc.client as client
import utils
from utils import load_model, load_image, preprocess_image, predict, publish_msg, predict_and_publish_msg, gstreamer_pipeline

if config_utils.ENABLE_SEND_MESSAGE:
    # Get the ipc client
    try:
        ipc = ipc_utils.IPCUtils()
        ipc_client = client.GreengrassCoreIPCClient(ipc.connect())
        config_utils.logger.info("Created IPC client...")
    except Exception as e:
        config_utils.logger.error(
            "Exception occured during the creation of an IPC client: {}".format(e)
        )
        exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AIoT Demo')
    parser.add_argument('--use_camera', 
                        default=0, type=int,
                        help='Whether to use Camera (default: 0')
    parser.add_argument('--width', 
                        default=1280, type=int,
                        help='Camera width (default: 1280')
    parser.add_argument('--height', 
                        default=720, type=int,
                        help='Camera height (default: 720')                    
    args = parser.parse_args()

    config_utils.logger.info(args)
    os.system("echo {}".format("Using dlr from '{}'.".format(dlr.__file__)))
    os.system("echo {}".format("Using numpy from '{}'.".format(np.__file__)))    

    if config_utils.USE_GPU:
        model = load_model(config_utils.MODEL_GPU_DIR, 'gpu')
    else:
        model = load_model(config_utils.MODEL_CPU_DIR, 'cpu')

    config_utils.logger.info('====== OpenCV Version Check ======\n')
    config_utils.logger.info(cv2.__version__)
    config_utils.logger.info(cv2.getBuildInformation())

    if args.use_camera == 1:
        utils.demo_camera(model, args)
    else:
        utils.demo_sample_images(model)