import os
import json
import random
import warnings
import logging
import sys
import train_utils
import copy
import time
import argparse
from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.distributed import Backend
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
logger = train_utils.set_logger()

        
def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Hyperparameter Setting
    parser.add_argument('--model_name', type=str, default='mobilenetv2')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)

    # SageMaker Container environment
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_chkpt_dir', type=str,
                        default='/opt/ml/checkpoints')    
    parser.add_argument('--train_dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--valid_dir', type=str,
                        default=os.environ['SM_CHANNEL_VALID'])    
    parser.add_argument('--num_gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def trainer(current_gpu, model, dataloaders, transforms, args):
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    feature_extract = False    
    
    optimizer = train_utils.initialize_optimizer(model, feature_extract, lr=1e-3, momentum=0.9)    
    criterion = nn.CrossEntropyLoss()

    # Send the model to GPU
    model = model.to(args.device)
    
    since = time.time()
    best_acc1 = 0.0

    num_samples = {k: len(dataloaders[k].dataset) for k, v in dataloaders.items()}
    num_steps = {k: int(np.ceil(len(dataloaders[k].dataset) / (batch_size))) for k, v in dataloaders.items()}

    for epoch in range(1, num_epochs+1):

        batch_time = train_utils.AverageMeter('Time', ':6.3f')
        data_time = train_utils.AverageMeter('Data', ':6.3f')
        losses = train_utils.AverageMeter('Loss', ':.4e')
        top1 = train_utils.AverageMeter('Acc@1', ':6.2f')

        logger.info('-' * 40)
        logger.info('[Epoch {}/{}] Processing...'.format(epoch, num_epochs))
        logger.info('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_num_samples = 0
            epoch_tic = time.time()            
            tic = time.time()        

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # measure data loading time
                data_time.update(time.time() - tic)

                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    probs, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_num_samples += inputs.size(0)
                
                acc1 = train_utils.accuracy(outputs, labels, topk=(1,)) 

                losses.update(train_utils.to_python_float(loss.data), inputs.size(0))
                top1.update(train_utils.to_python_float(acc1[0]), inputs.size(0))
                batch_time.update(time.time() - tic)
                tic = time.time()

                if phase == 'train' and i % args.log_interval == 1:
                    step_loss = running_loss / running_num_samples
                    step_acc = running_corrects.double() / running_num_samples
                    logger.info(f'[Epoch {epoch}/{num_epochs}, Step {i+1}/{num_steps[phase]}] {phase}-acc: {step_acc:.4f}, '
                             f'{phase}-loss: {step_loss:.4f}, data-time: {data_time.val:.4f}, batch-time: {batch_time.val:.4f}')            
            logger.info(f'[Epoch {epoch}/{num_epochs}] {phase}-acc: {top1.avg:.4f}, '
                         f'{phase}-loss: {losses.val:.4f}, time: {time.time()-epoch_tic:.4f}') 

            if phase == 'valid':
                is_best = top1.avg > best_acc1
                best_acc1 = max(top1.avg, best_acc1)

                train_utils.save_model({
                    'epoch': epoch + 1,
                    'model_name': args.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                    'loss': losses
                }, is_best, args.model_chkpt_dir, args.model_dir)                 

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val acc: {:.4f}'.format(best_acc1))    
    
    
if __name__ == '__main__':
    
    is_sm_container = True    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False
        
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        os.environ['SM_CURRENT_HOST'] = 'algo-1'
        os.environ['SM_HOSTS'] = json.dumps(["algo-1"])
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/model'
        os.environ['SM_NUM_GPUS'] = str(1)
        dataset_dir = f'{src_dir}/smartfactory'
        os.environ['SM_CHANNEL_TRAIN'] = f'{dataset_dir}/train'
        os.environ['SM_CHANNEL_VALID'] = f'{dataset_dir}/valid'  
        
    args = parser_args()
    args.use_cuda = args.num_gpus > 0
    
    print("args.use_cuda : {} , args.num_gpus : {}".format(
        args.use_cuda, args.num_gpus))
    args.kwargs = {'pin_memory': True} if args.use_cuda else {}
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args.rank = 0
    args.world_size = 1
    
    os.makedirs(args.model_chkpt_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    args.classes, args.classes_dict = train_utils.get_classes(args.train_dir) 
    args.num_classes = len(args.classes)
    
    dataloaders, transforms, train_sampler = train_utils.create_dataloaders(
        args.train_dir, args.valid_dir, rank=args.rank, 
        world_size=args.world_size, batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    feature_extract = False
    model = train_utils.initialize_ft_model(args.model_name, num_classes=args.num_classes, feature_extract=feature_extract)

    trainer(0, model, dataloaders, transforms, args)
