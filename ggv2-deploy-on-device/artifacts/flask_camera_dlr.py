import logging
import platform
import sys
import threading
import time
import os
import argparse
import cv2
import numpy as np
import config_utils
from flask import Flask, render_template, Response, request
from utils import load_model, load_image, preprocess_image, predict, publish_msg, predict_and_publish_msg, gstreamer_pipeline
from datetime import datetime
from threading import Thread

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

global capture
capture = 0
os.makedirs('screenshot', exist_ok=True)

if config_utils.USE_GPU:
    model = load_model(config_utils.MODEL_GPU_DIR, 'gpu')   
else:
    model = load_model(config_utils.MODEL_CPU_DIR, 'cpu')   

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST','GET'])
def tasks():
    global switch,cap
    if request.method == 'POST':
        if request.form.get('capture') == 'Capture':
            global capture
            capture = 1                         
                 
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


def gen_frames():  # generate frame by frame from camera
     # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    global cap
    global capture
    print(gstreamer_pipeline(flip_method=0))
    prev_t = 0 
    
    while(cap.isOpened()):        
         
        # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame

        if not success:
            break
        else:
            ########################################################
            # ML Prediction
            ########################################################
            # Prediction
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_class, pred_str, prob, msg = predict(img_rgb, model)

            # Compute FPS
            curr_t = time.time()
            fps = 1./(curr_t - prev_t)
            prev_t = curr_t         
            
            # Display prediction result
            #put_msg = f'{msg}, fps={fps:.2f}'
            put_msg = f'{msg}'
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, put_msg, (30,40), font, 1, (150,255,0), 2, cv2.LINE_AA)

            # Puiblish MQTT message
            payload = publish_msg(pred_class, pred_str, prob)

            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == ord('q'):
                break

            ########################################################
            # Capture
            ########################################################
            if capture:
                capture = 0
                write_path = os.path.sep.join(['screenshot', "img_{}.png".format(str(datetime.now()).replace(":",''))])
                cv2.imwrite(write_path, frame)

            ########################################################
            # Send to HTML 
            ########################################################
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    cap.release()
    cv2.destroyAllWindows()


def response_web_inference(ip_addr, port):
    if ip_addr is not None:
        app.run(host=ip_addr, port=port, debug=False)
    else:
        logger.error("[ERROR]------NO IP ADDRESS--------")


def start_app(ip_addr, port):
    # message_thread = threading.Thread(target=publish_inference_result)
    web_thread = threading.Thread(target=response_web_inference, args=(ip_addr, port))

    # message_thread.start()
    web_thread.start()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AIoT Demo')
    parser.add_argument('--use_gpu', 
                        default=0, type=int,
                        help='Use GPU (default: 0')    
    parser.add_argument('--ip_addr', 
                        type=str,
                        help='IP Address')                          
    parser.add_argument('--port', 
                        type=int,
                        help='Port number (default: 1234')                          
    args = parser.parse_args()
    config_utils.logger.info(args)

    logger.info('-------------START--------------')
    logger.info('---IP Address : {}'.format(args.ip_addr))
    logger.info('---Port Number : {}'.format(args.port))
    logger.info('--------------------------------')

    start_app(ip_addr=args.ip_addr, port=args.port)
