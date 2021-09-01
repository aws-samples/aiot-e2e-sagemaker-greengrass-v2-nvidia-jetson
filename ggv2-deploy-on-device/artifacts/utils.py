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


def publish_msg(pred_class, pred_str, prob, verbose=0):

    message = '{"class_id":"' + str(pred_class) + '"' + ',"class":"' + pred_str + '"' + ',"score":"' + str(prob) +'"}'
    payload = {
        "message": message,
        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    }
    config_utils.logger.info(payload)    
    
    if config_utils.ENABLE_SEND_MESSAGE:
        ipc.publish_results_to_cloud(ipc_client, payload)
        
    if verbose == 1:        
        print(json.dumps(payload, sort_keys=True, indent=4))        

    return payload


def predict_and_publish_msg(img_filepath, model, verbose=0):

    img = cv2.imread(img_filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_class, pred_str, prob, msg = predict(img_rgb, model)

    payload = publish_msg(pred_class, pred_str, prob, verbose)

    return payload


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def demo_camera(model, args):
    gstreamer = gstreamer_pipeline(
        capture_width=args.width, capture_height=args.height, 
        display_width=args.width, display_height=args.height
    )
    print(gstreamer)    
    cap = cv2.VideoCapture(gstreamer, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        prev_t = 0        
        #window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
 
        while True:
        #while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()

            # Prediction
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_class, pred_str, prob, msg = predict(img_rgb, model)

            # Compute FPS
            curr_t = time.time()
            fps = 1./(curr_t - prev_t)
            prev_t = curr_t            

            # Display prediction result
            put_msg = f'{msg}, fps={fps:.2f}'
            font = cv2.FONT_HERSHEY_COMPLEX
            #cv2.putText(img, put_msg, (30,40), font, 1, (150,255,0), 2, cv2.LINE_AA)
            #cv2.imshow("CSI Camera", img)

            # Puiblish MQTT message
            payload = publish_msg(pred_class, pred_str, prob)

            # Stop the program on the ESC key
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


def demo_sample_images(model):
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
        payload = predict_and_publish_msg(img_filepath, model)
        
        idx += 1
        if idx % num_imgs == 0:
            idx = 0
            print(idx)

        time.sleep(config_utils.DEFAULT_PREDICTION_INTERVAL_SECS)
