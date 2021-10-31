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
from utils import load_model, load_image, preprocess_image, predict, publish_msg, predict_and_publish_msg

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
else:
    ipc = None; ipc_client = None

if __name__ == "__main__":

    os.system("echo {}".format("Using dlr from '{}'.".format(dlr.__file__)))
    os.system("echo {}".format("Using numpy from '{}'.".format(np.__file__)))    

    if config_utils.USE_GPU:
        model = load_model(config_utils.MODEL_GPU_DIR, 'gpu')
    else:
        model = load_model(config_utils.MODEL_CPU_DIR, 'cpu')

    config_utils.logger.info('====== OpenCV Version Check ======\n')
    config_utils.logger.info(cv2.__version__)
    config_utils.logger.info(cv2.getBuildInformation())
    

    extensions = (f"{config_utils.SAMPLE_IMAGE_DIR}/*.jpg", f"{config_utils.SAMPLE_IMAGE_DIR}/*.jpeg")
    img_filelist = [f for f_ in [glob.glob(e) for e in extensions] for f in f_]
    config_utils.logger.info(img_filelist)

    idx = 0
    num_imgs = len(img_filelist)

    while True:
        # Prepare image
        img_filepath = img_filelist[idx]
        config_utils.logger.info(f'\n image path = {img_filepath}')
        # Predict
        payload = predict_and_publish_msg(ipc, ipc_client, img_filepath, model)
        
        idx += 1
        if idx % num_imgs == 0:
            idx = 0
            print(idx)

        time.sleep(config_utils.DEFAULT_PREDICTION_INTERVAL_SECS)
        