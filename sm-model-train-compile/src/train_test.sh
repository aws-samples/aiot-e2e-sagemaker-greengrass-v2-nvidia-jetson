sudo mkdir -p /opt/ml/model
sudo mkdir -p /opt/ml/checkpoints
sudo chown ec2-user:ec2-user -R /opt/ml/model
sudo chown ec2-user:ec2-user -R /opt/ml/checkpoints

python train_multi_gpu.py