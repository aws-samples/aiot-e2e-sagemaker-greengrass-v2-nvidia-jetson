#!/bin/bash

fullcmd="$(realpath $0)" 
thispath="$(dirname $fullcmd)"

echo '====== Component Folder ======' 
echo $fullcmd
echo $thispath

cd $thispath
echo '====== Install Dependencies ======' 

python3 -m venv
source gcv/bin/activate

pip3 install pip --upgrade
pip3 install -r requirements.txt
