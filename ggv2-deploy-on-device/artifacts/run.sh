#!/bin/bash

fullcmd="$(realpath $0)" 
thispath="$(dirname $fullcmd)"
cd $thispath
source gcv/bin/activate

while getopts ":c:" o ; do
    case "${o}" in 
        c)
            use_camera=${OPTARG}
            ;;
    esac
done

python3 inference.py --use_camera=$use_camera 