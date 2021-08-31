#!/bin/bash

fullcmd="$(realpath $0)" 
thispath="$(dirname $fullcmd)"
cd $thispath
source gcv/bin/activate

while getopts ":i:p:" o ; do
    case "${o}" in
        i)
            ip_addr=${OPTARG}
            ;;
        p)
            port=${OPTARG}
            ;;         
    esac
done

python3 flask_camera_dlr.py --ip_addr=$ip_addr --port=$port