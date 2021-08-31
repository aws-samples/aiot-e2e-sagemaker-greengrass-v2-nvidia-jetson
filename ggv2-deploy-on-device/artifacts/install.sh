#!/bin/bash

fullcmd="$(realpath $0)" 
thispath="$(dirname $fullcmd)"

echo '====== Component Folder ======' 
echo $fullcmd
echo $thispath

cd $thispath
echo '====== Install Dependencies ======' 

python3 -m venv gcv
source gcv/bin/activate

pip3 install pip --upgrade
pip3 install -r requirements.txt

echo '====== Install DLR ======' 
cd packages
pip3 install dlr-1.9.0-py3-none-any.whl
pip3 install opencv_python-4.5.3+c1cc7e5-cp36-cp36m-linux_aarch64.whl

#echo "===== Importing opencv library from host into virtualenv ====="
#cp /usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so /home/ggc_user/.local/lib/python3.6/site-packages/cv2