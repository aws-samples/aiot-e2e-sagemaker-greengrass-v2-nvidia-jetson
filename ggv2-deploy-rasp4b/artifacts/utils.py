import os
import dlr
import cv2
import numpy as np
from dlr import DLRModel

import argparse
import time
import glob
from datetime import datetime
import json
import config_utils
import IPCUtils as ipc_utils


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    f_x = x_exp / np.sum(x_exp)
    return f_x


def load_model(model_dir, device):
    model = DLRModel(model_dir, device)
    return model


def load_image(image_path):
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    return image_data


def preprocess_image(image):
    cvimage = cv2.resize(image, config_utils.SHAPE)
    #config_utils.logger.info("img shape after resize: '{}'.".format(cvimage.shape))
    img = np.asarray(cvimage, dtype='float32')
    img /= 255.0 # scale 0 to 1
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1)) 
    img = np.expand_dims(img, axis=0) # e.g., [1x3x224x224]
    return img


def predict(image, model):
    image_data = preprocess_image(image)
    output = model.run(image_data)  
    probs = softmax(output[0][0])
    sort_classes_by_probs = np.argsort(probs)[::-1]

    max_no_of_results = 1
    pred_class = sort_classes_by_probs[0]
    prob = probs[pred_class]
    pred_str = config_utils.LABEL_MAP[pred_class]
    msg = f'{config_utils.LABEL_MAP[pred_class]}, {probs[pred_class]*100:.2f}%'    
    return pred_class, pred_str, prob, msg


def publish_msg(ipc, ipc_client, pred_class, pred_str, prob, verbose=0):
    message = '{"class_id":"' + str(pred_class) + '"' + ',"class":"' + pred_str + '"' + ',"score":"' + str(prob) +'"}'
    payload = {
        "message": message,
        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    }
    config_utils.logger.info(payload)    
    
    if config_utils.ENABLE_SEND_MESSAGE and ipc_client is not None:
        ipc.publish_results_to_cloud(ipc_client, payload)
        
    if verbose == 1:        
        print(json.dumps(payload, sort_keys=True, indent=4))        

    return payload


def predict_and_publish_msg(ipc, ipc_client, img_filepath, model, verbose=0):
    img_rgb = load_image(img_filepath)
    pred_class, pred_str, prob, msg = predict(img_rgb, model)
    payload = publish_msg(ipc, ipc_client, pred_class, pred_str, prob, verbose)
    return payload
