#!/bin/bash

fullcmd="$(realpath $0)" 
thispath="$(dirname $fullcmd)"
cd $thispath
source gcv/bin/activate

python3 inference.py 