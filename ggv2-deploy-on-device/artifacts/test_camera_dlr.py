import os
import argparse
import dlr
import cv2
import numpy as np
from dlr import DLRModel
import config_utils
import time
from utils import load_model, load_image, preprocess_image, predict, publish_msg, gstreamer_pipeline


def show_camera(model, args):
    gstreamer = gstreamer_pipeline(
        capture_width=args.width, capture_height=args.height, 
        display_width=args.width, display_height=args.height
    )
    print(gstreamer)
    cap = cv2.VideoCapture(gstreamer, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        prev_t = 0            
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()

            # Prediction
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_class, pred_str, prob, msg = predict(img_rgb, model)

            # Compute FPS
            curr_t = time.time()
            fps = 1./(curr_t - prev_t)
            prev_t = curr_t

            # Show results
            put_msg = f'{msg}, fps={fps:.2f}'
            font = cv2.FONT_HERSHEY_COMPLEX            
            cv2.putText(img, put_msg, (30,40), font, 1, (150,255,0), 2, cv2.LINE_AA)
            cv2.imshow("CSI Camera", img)

            # Stop the program on the ESC key
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AIoT Demo')
    parser.add_argument('--width', 
                        default=1280, type=int,
                        help='Camera width (default: 1280')
    parser.add_argument('--height', 
                        default=720, type=int,
                        help='Camera height (default: 720')

    args = parser.parse_args()
    config_utils.logger.info(args)    

    if config_utils.USE_GPU:
        model = load_model(model_dir='model_gpu', device='gpu')
    else:
        model = load_model(model_dir='model_cpu', device='cpu')            
    
    show_camera(model, args)
