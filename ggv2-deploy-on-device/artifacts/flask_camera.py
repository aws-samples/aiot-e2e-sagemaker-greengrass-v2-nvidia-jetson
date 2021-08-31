import logging
import platform
import sys
import threading
import time
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response


SW_VERSION = 'ver-1.0.1'
IP_ADDRESS = os.environ['DEVICE_IP']
PORT_NUMBER = os.environ.get('PORT_NUMBER', '1234')
TOPIC_INFERENCE = "aws/inference"
MSG_ERROR_NO_IPADDRESS = '[ERROR]------NO IP ADDRESS--------'

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger.info('-------------START--------------')
logger.info('---SW Version : {}'.format(SW_VERSION))
logger.info('---IP Address : {}'.format(IP_ADDRESS))
logger.info('---Port Number : {}'.format(PORT_NUMBER))
logger.info('---Inference Topic : {}'.format(TOPIC_INFERENCE))
logger.info('--------------------------------')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def camera_convert(frame):
	return np.transpose(frame.reshape(3, 480, 640), (1, 2, 0))


def gen_frames():  # generate frame by frame from camera

    # To flip the image, modify the flip_method parameter (0 and 2 are the
    # most common)
    print(gstreamer_pipeline(flip_method=0))

    cap = cv2.VideoCapture(
        gstreamer_pipeline(
            flip_method=0),
        cv2.CAP_GSTREAMER)

    while(cap.isOpened()):        
        # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def response_web_inference():
    if IP_ADDRESS is not None:
        app.run(host=IP_ADDRESS, port=PORT_NUMBER, debug=False)
    else:
        logger.error(MSG_ERROR_NO_IPADDRESS)


def start_app():
    # message_thread = threading.Thread(target=publish_inference_result)
    web_thread = threading.Thread(target=response_web_inference)

    # message_thread.start()
    web_thread.start()


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

if __name__ == "__main__":
    start_app()
