#!/bin/bash

currPath="$(pwd)" 
projectPath="$(dirname $currPath)/$1"

sudo rm -rf /opt/ml/model /opt/ml/checkpoints
sudo mkdir -p /opt/ml/model
sudo mkdir -p /opt/ml/checkpoints
sudo chown ec2-user:ec2-user -R /opt/ml/model
sudo chown ec2-user:ec2-user -R /opt/ml/checkpoints

python3 train_multi_gpu.py --train_dir=$projectPath/train --valid_dir=$projectPath/valid --num_epochs=8