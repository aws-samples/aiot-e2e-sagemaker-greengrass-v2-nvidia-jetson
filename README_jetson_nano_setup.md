# How to Run DLR and Camera on NVIDIA Jetson Nano Jetpack 4.5.1

### Author: Daekeun Kim (daekeun@)

The instructions below assume that you have installed Jetpack 4.5.1. But even if the jetpack version is updated, the installation method won't change much.

## 1. Prerequisites and Dependencies
```bash
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install libopenblas-dev

# NVIDIA Jetpack's Python does not have built-in venv and pip by default
sudo apt-get install python3-venv python3-pip libgoogle-glog-devsudo 
pip3 install -U pip testresources setuptools
```

## 2. Install and check Jetson Stats
```bash
sudo -H pip3 install -U jetson-stats
jetson_release 
sudo jtop
```

## 3. Build CMake 
It is not required if you install it as Option 1 in the Build DLR section. However, other packages may need to be compiled in the future, so it is recommended to install them beforehand.
```bash
sudo apt-get install libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz
tar xvf cmake-3.17.2.tar.gz
cd cmake-3.17.2
./bootstrap
make -j4 # recommend make -j3
sudo make install
```

## 4. Install numpy & libcanberra
[Note] Please do not use the latest version of numpy, be sure to specify 1.19.x, such as 1.19.4 and 1.19.5. Using the latest version(>=1.2.0) throws a Python 3.7 dependency error when installing DLR.

```bash
pip3 install protobuf>=3.3.0 Cython
pip3 install numpy==1.19.4
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
```

## 5. Build DLR 
Please choose one of three options. Option 1 is recommended.

### Option 1: Pre-built wheel for NVIDIA Jetson Nano (~1 min, It was built by the author)
- [Download](ggv2-deploy-on-device/artifacts/packages/dlr-1.9.0-py3-none-any.whl) `ggv2-deploy-on-device/artifacts/packages/dlr-1.9.0-py3-none-any.whl`.
- Run `pip3 install dlr-1.9.0-py3-none-any.whl` on your device.
 
```bash
unzip dlr_1.9.0_jetson_nano.zip
cd dlr_1.9.0_jetson_nano/python
python3 setup.py install --user
#sudo python3 setup.py install #If you want to install globally
```

### Option 2: Build From Scratch (~30 mins)
```bash
git clone --recursive https://github.com/neo-ai/neo-ai-dlr
cd neo-ai-dlr && mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=ON
make -j4 # recommend make -j3
cd ../python && python3 setup.py install --user
```

## 6. (Optional, but Recommend if you need high-performance) Install OpenCV

- See https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html

## 7. Check GStreamer Camera on OpenCV

### References:
- https://github.com/JetsonHacksNano/CSI-Camera
- https://velog.io/@devseunggwan/Vision-Nvidia-Jetson-Nano%EC%97%90%EC%84%9C-CSI-Camera-%EC%82%AC%EC%9A%A9-%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%B6%95-%EB%B0%8F-%ED%85%8C%EC%8A%A4%ED%8A%B8 (Korean)

### Check GStreamer is working
```bash
gst-launch-1.0 nvcamerasrc auto-exposure=1 exposure-time=.0005 ! nvoverlaysink
```

### Check OpenCV
- Save the code below as simple_camera.py and run python3 simple_camera.py,
    - simple_camera.py (https://raw.githubusercontent.com/JetsonHacksNano/CSI-Camera/master/simple_camera.py)
```python
# MIT License≈
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


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


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
```

- If the camera image is not displayed on the monitor and an `‘Unable to open camera error’` is displayed on the console, please reinstall OpenCV.
  - If the Gstreamer command works normally, but an error occurs in `cv2.VideoCapture(...)` of OpenCV, meaning that the Gstreamer option is not activated in OpenCV with a nearly 100% probability.
  - As of Jetpack 4.5.1, the built-in OpenCV 4.1.1's GStreamer option is enabled, but unfortunately CUDA is not enabled, resulting in performance degradation. Therefore, OpenCV must be recompiled to take full advantage of OpenCV.
  - Please never install with `pip3 install opencv-python` on jetson nano! The `opencv-python` wheel package makes the camera unusable in OpenCV because the GStreamer option is off. It is time consuming, but it is recommended to compile from scratch.
  
## 8. Optional

### Install Jetson Fan Control 
```bash
git clone https://github.com/Pyrestone/jetson-fan-ctl.gi
cd jetson-fan-ctl 
sudo ./install.sh
```

### Install VSCode
```bash
git clone https://github.com/JetsonHacksNano/installVSCode.git
cd installVSCode/
./installVSCode.sh
```
