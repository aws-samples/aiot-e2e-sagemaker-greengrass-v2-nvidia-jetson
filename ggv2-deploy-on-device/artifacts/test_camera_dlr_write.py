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
        cnt = 0
        prev_t = 0        
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read() 
            img_clone = img.copy()

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

            if cnt < args.max_write_img:
                img_name = f'{args.raw_img_dir}/img_{cnt:06d}.jpg'
                cv2.imwrite(img_name, img_clone)
            cnt+=1   

            # Stop the program on the ESC key
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AIoT Demo')
    parser.add_argument('--width', 
                        default=1280, type=int,
                        help='Camera width (default: 1280')
    parser.add_argument('--height', 
                        default=720, type=int,
                        help='Camera height (default: 720')    
    parser.add_argument('--raw_img_dir', 
                        default='../rawimg', type=str,
                        help='Raw Image Directory (default: ../rawimg')
    parser.add_argument('--max_write_img', 
                        default=10, type=int,
                        help='Maximum Raw Images (default: 10')
    args = parser.parse_args()
    config_utils.logger.info(args)
    os.makedirs(f'{args.raw_img_dir}', exist_ok=True)

    model = load_model(model_dir='model_cpu', device='cpu')
    show_camera(model, args)