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
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
logger = train_utils.set_logger()

        
def dist_init(fn, args):
    """
    Initialize the distributed environment by spawning multiple processes.
    It sets up a different distributed environment for each process.
    """
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

    args.is_distributed = len(args.hosts) > 1 and args.backend is not None
    args.is_multigpus = args.num_gpus > 1
    args.multigpus_distributed = (args.is_distributed or args.is_multigpus)

    logger.debug("Distributed training - {}".format(args.is_distributed))
    logger.debug("Number of gpus available - {}".format(args.num_gpus))

    args.world_size = 1
    if args.multigpus_distributed:
        args.world_size = len(args.hosts) * args.num_gpus
        os.environ['WORLD_SIZE'] = str(args.world_size)
        args.host_num = args.hosts.index(args.current_host)
        mp.spawn(fn, 
                 nprocs=args.num_gpus, 
                 args=(args,))
    else:
        current_gpu = 0
        fn(current_gpu, args) 

    
def dist_setting(current_gpu, model, args):
    """
    Perform basic settings for distributed training and initializes the process group (dist.init_process_group).
    """
    print("channels_last : {}".format(args.channels_last))
    if args.channels_last:
        args.memory_format = torch.channels_last
    else:
        args.memory_format = torch.contiguous_format

    if args.apex:
        args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    args.current_gpu = current_gpu
    if args.current_gpu is not None:
        print("Use GPU: {} for training".format(args.current_gpu))

    if args.multigpus_distributed:
        args.rank = args.num_gpus * args.host_num + args.current_gpu
        dist.init_process_group(backend=args.backend,
                                rank=args.rank, world_size=args.world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))
    else:
        args.rank = 0

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if args.multigpus_distributed:
        if args.current_gpu is not None:
            torch.cuda.set_device(args.current_gpu)
            args.batch_size = int(args.batch_size / args.num_gpus)
            logger.info("Batch size for each GPU: {}".format(args.batch_size))
            if not args.apex:
                model.cuda(args.current_gpu)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.current_gpu])
        else:
            if not args.apex:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.current_gpu is not None:
        torch.cuda.set_device(args.current_gpu)
        if not args.apex:
            model = model.cuda(args.current_gpu)
    else:
        if not args.apex:
            model = torch.nn.DataParallel(model).cuda()

    return model, args


def dist_cleanup():
    """
    Deinitialize processes using torch.distributed.destroy_process_group
    """
    dist.destroy_process_group()
    
    
def trainer(current_gpu, args):
    """
    This is the training main function. Works on single GPU, multi-gpu, and multi-gpu/multi-instance. 
    But if you want to see a simple script to train on a single GPU, please check train_single_gpu.py.
    """

    model_history = train_utils.init_model_history()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    feature_extract = False

    model = train_utils.initialize_ft_model(args.model_name, num_classes=args.num_classes, feature_extract=feature_extract)
    model, args = dist_setting(current_gpu, model, args)
    logger.info(f"==> Training on rank {args.rank}.")
    logger.info(args)
    
    dataloaders, transforms, train_sampler = train_utils.create_dataloaders(
        args.train_dir, args.valid_dir, rank=args.rank, 
        world_size=args.world_size, batch_size=batch_size,
        num_workers=args.num_workers
    )
        
    optimizer = train_utils.initialize_optimizer(model, feature_extract, lr=args.lr*args.world_size, momentum=0.9)    
    criterion = nn.CrossEntropyLoss()

    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc1 = 0.0
    num_samples = {k: len(dataloaders[k].dataset) for k, v in dataloaders.items()}
    num_steps = {k: int(np.ceil(len(dataloaders[k].dataset) / (batch_size*args.world_size))) for k, v in dataloaders.items()}
    device = torch.device(f'cuda:{current_gpu}')    

    for epoch in range(1, num_epochs+1):
        
        batch_time = train_utils.AverageMeter('Time', ':6.3f')
        data_time = train_utils.AverageMeter('Data', ':6.3f')
        losses = train_utils.AverageMeter('Loss', ':.4e')
        top1 = train_utils.AverageMeter('Acc@1', ':6.2f')
        
        logger.info('-' * 40)
        logger.info('[Rank {}, Epoch {}/{}] Processing...'.format(args.rank, epoch, num_epochs))
        logger.info('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:        

            if phase == 'train':
                model.train()  # Set model to training mode
                if args.multigpus_distributed:
                    dataloaders[phase].sampler.set_epoch(epoch)  # Set epoch count for DistributedSampler          
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            epoch_tic = time.time()            
            tic = time.time()
            
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # measure data loading time
                data_time.update(time.time() - tic)                
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    probs, preds = torch.max(outputs, 1)
                    
                    # Compute gradient and do stochastic gradient descent
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                acc1 = train_utils.accuracy(outputs, labels, topk=(1,))
                
                # Average loss and accuracy across processes for logging
                if args.multigpus_distributed:
                    reduced_loss = train_utils.reduce_tensor(loss.data, args)
                    reduced_acc1 = train_utils.reduce_tensor(acc1[0], args)
                else:
                    reduced_loss = loss.data
                    reduced_acc1 = acc1[0]

                losses.update(train_utils.to_python_float(reduced_loss), inputs.size(0))
                top1.update(train_utils.to_python_float(reduced_acc1), inputs.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - tic)
                tic = time.time()

                if phase == 'train' and i % args.log_interval == 0:
                    step_loss = running_loss / ((i+1)*batch_size)
                    step_acc = running_corrects.double() / ((i+1)*batch_size)
                    logger.info(f'[Rank {args.rank}, Epoch {epoch}/{num_epochs}, Step {i+1}/{num_steps[phase]}] {phase}-acc: {step_acc:.4f}, '
                             f'{phase}-loss: {step_loss:.4f}, data-time: {data_time.val:.4f}, batch-time: {batch_time.val:.4f}')           
                    

            ## Waiting until finishing operations on GPU (Pytorch default: async)
            torch.cuda.synchronize()
        
            if current_gpu == 0:    
                logger.info(f'[Epoch {epoch}/{num_epochs}] {phase}-acc: {top1.avg:.4f}, '
                             f'{phase}-loss: {losses.val:.4f}, time: {time.time()-epoch_tic:.4f}')  
                
                model_history['epoch'].append(epoch)
                model_history['batch_idx'].append(i)
                model_history['data_time'].append(data_time.val)                
                model_history['batch_time'].append(batch_time.val)
                model_history['losses'].append(losses.val)
                model_history['top1'].append(top1.val)

            if phase == 'valid':
                is_best = top1.avg > best_acc1
                best_acc1 = max(top1.avg, best_acc1)
    
                if (args.multigpus_distributed and args.rank % args.num_gpus == 0):
                    #train_utils.save_history(os.path.join(args.output_data_dir, 'model_history.p'), model_history) 
                    train_utils.save_model({
                        'epoch': epoch + 1,
                        'model_name': args.model_name,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc1': best_acc1,
                        'loss': losses
                    }, is_best, args.model_chkpt_dir, args.model_dir)
                elif not args.multigpus_distributed:
                    #train_utils.save_history(os.path.join(args.output_data_dir, 'model_history.p'), model_history) 
                    train_utils.save_model({
                        'epoch': epoch + 1,
                        'model_name': args.model_name,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc1': best_acc1,
                        'loss': losses
                    }, is_best, args.model_chkpt_dir, args.model_dir)                    
                    
        
    time_elapsed = time.time() - since
    if current_gpu == 0:
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best val acc: {:.4f}'.format(best_acc1))
    
    if args.multigpus_distributed:
        dist_cleanup()   
        
        
def parser_args():
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--channels_last', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # Hyperparameter Setting
    parser.add_argument('--model_name', type=str, default='mobilenetv2')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('--sync_bn', action='store_true', 
                        help='enabling apex sync BN.', default=False)

    os.environ.get('HOME', '/home/username/')
    
    # SageMaker Container environment
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ.get('SM_HOSTS', '["algo-1"]')))
    parser.add_argument('--current_host', type=str,
                        default=os.environ.get('SM_CURRENT_HOST', 'algo-1'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--model_chkpt_dir', type=str,
                        default='/opt/ml/checkpoints')    
    parser.add_argument('--train_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/train'))
    parser.add_argument('--valid_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_VALID', '/opt/ml/valid'))    
    parser.add_argument('--num_gpus', type=int,
                        default=os.environ.get('SM_NUM_GPUS', str(torch.cuda.device_count())))
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()
    return args

    
if __name__ == '__main__': 

    args = parser_args()
    args.use_cuda = args.num_gpus > 0
    
    print("args.use_cuda : {} , args.num_gpus : {}".format(
        args.use_cuda, args.num_gpus))
    args.kwargs = {'pin_memory': True} if args.use_cuda else {}
    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    current_host = args.current_host
    hosts = args.hosts
    host = hosts.index(current_host)
    print(current_host)
    print(hosts)
    print(host)
    
    os.makedirs(args.model_chkpt_dir, exist_ok=True)
    print(args)
    args.classes, args.classes_dict = train_utils.get_classes(args.train_dir) 
    args.num_classes = len(args.classes)

    dist_init(trainer, args)
    