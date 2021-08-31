import copy
import time
import numpy as np
import torch, os
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from typing import Tuple
import logging
import sys
import codecs
import json
import shutil


def set_logger():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )

    logger = logging.getLogger(__name__)
    return logger


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_classes(train_path):
    #https://github.com/pytorch/vision/blob/50d9dc5f5af89e607100cee9aa34cfda67e627fb/torchvision/datasets/folder.py#L114
    classes = [d.name for d in os.scandir(train_path) if d.is_dir()]
    classes.sort()
    classes_dict = {i:c for i, c in enumerate(classes)}
    return classes, classes_dict
    
    
def save_classes_dict(classes_dict, filename='classes_dict.json'):
    with open(filename, "w") as fp:
        json.dump(classes_dict, fp) 
        
        
def load_classes_dict(filename):
    with open(filename, 'r') as f:
        classes_dict = json.load(f)
        
    classes_dict = {int(k):v for k, v in classes_dict.items()}
    return classes_dict


def create_transforms():
    train_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        #T.RandomResizedCrop(224),
        #T.RandomHorizontalFlip(p=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = T.Compose([
        T.Resize(224),
        #T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transforms = {
        'train': train_transform,
        'valid': valid_transform
    }
    
    return transforms


def create_inv_transform():
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    ) 
    return inv_normalize


def create_dataloaders(train_path: str, valid_path: str, 
                       rank: int, world_size: int, 
                       batch_size: int, num_workers: int=4) -> Tuple[dict, dict]:

    transforms = create_transforms()
    
    train_dataset = ImageFolder(root=train_path, transform=transforms['train'], target_transform=None)
    valid_dataset = ImageFolder(root=valid_path, transform=transforms['valid'], target_transform=None)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=int(rank)) if world_size > 1 else None
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=int(rank)) if world_size > 1 else None
    
    # shuffling is done by DistributedSampler    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_sampler is None, 
                                  sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                  sampler=valid_sampler, num_workers=num_workers, pin_memory=True)

    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
    }
    
    return dataloaders, transforms, train_sampler


def visualize_dataloader_samples(dataloder, classes, nrow=4, need_denormalize=True):
    dataiter = iter(dataloder)
    imgs, labels = dataiter.next() 
    batch_size = len(labels)
    fig_size = int(batch_size / nrow)
    img = torchvision.utils.make_grid(imgs, nrow=nrow)

    if need_denormalize:   
        inv_normalize = create_inv_transform()
        img = inv_normalize(img)
    
    img_np = np.transpose(img.numpy(), (1,2,0))
    img_np = np.clip(img_np, 0, 1) 
    plt.figure(figsize = (16,2*fig_size))
    plt.imshow(img_np)
    
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_ft_model(model_name, num_classes, feature_extract=True, use_pretrained=True):

    model_ft = None

    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "mobilenetv2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)       
   
    elif model_name == "mnasnet":
        model_ft = models.mnasnet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft    



def initialize_optimizer(model, feature_extract, lr=2e-5, momentum=0.9):
    
    params_to_update = model.parameters()
    print("=== Params to learn ===")

    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print(name)

    # Observe that all parameters are being optimized
    #optimizer = optim.AdamW(params_to_update, lr=0.0005, betas=(0.9, 0.999), eps=1e-08)
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)

    return optimizer


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    rt /= args.world_size
    return rt

    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
            #res.append(correct_k.mul_(100.0 / batch_size))
        return res

    
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

        
def init_model_history():
    model_history = {}
    model_history['epoch'] = []
    model_history['batch_idx'] = []
    model_history['data_time'] = []   
    model_history['batch_time'] = []
    model_history['losses'] = []
    model_history['top1'] = []
    return model_history


def save_history(path, history):
    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in history.keys():
        history_for_json[key] = list(map(float, history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(
            ',', ':'), sort_keys=True, indent=4)    
        

def save_model(state, is_best, model_chkpt_dir, model_dir):
    print("Saving the model.")
    filename = os.path.join(model_chkpt_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))