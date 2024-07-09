---
layout: distill
title: Pytorch Basic Code
date: 2024-07-08 11:00:00
description: Dataset, DataLoader, Train, ...
tags: pytorch
categories: cv-tasks
# thumbnail: assets/img/2024-07-08-PytorchBasic/1.png
giscus_comments: true
related_posts: true
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Pytorch Basic Code

### Deal with json, csv

```Python
import os
import json
import pandas as pd
import csv

# read json
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    f.close()

# read csv 방법 1. general case
data = pd.read_csv(csv_path, sep="|", index_col=0, skiprows=[1], na_values=['?', 'nan']).values # 0-th column (1-th row는 제외) ('?'와 'nan'은 결측값으로 인식)
# read csv 방법 2. special case : csv가 row별로 dictionary 형태일 때
if os.path.exists(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        data = [{key : value for key, value in row.items()} for row in reader] # row별로 읽음

# write to csv
dataset = [] # list of dictionaries
dataset.append({"id":id, "w":w, "h":h, "class":i})
pd.DataFrame(dataset).to_csv(output_path, index=False) # output_path : ".../dataset.csv"
```

### Create Dataset

```Python
import torch
from torch.utils.data import Dataset

# Create Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        super(CustomDataset, self).__init__()
        self.x = #...
        self.y = #...
        # 만약 load할 data가 너무 크다면 __init__()에서는 load할 파일명만 저장해놓고 __getitem__()에서 필요할 때마다 load

    def __getitem__(self, index): # should return float tensor
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
```

### DataLoader

- `rank` :  
  - 전체 distributed system에서 process 순서  
  - 4-CPU system이 2개 있을 경우  
  rank = machine 번호(0~1) * machine 당 process 개수(4) + process 번호(0~3)  
  - rank = 0인 process에 대해서만 wandb로 train log 출력

- `world_size` :  
  - 전체 distributed system에서 총 process 개수  
  - 4-GPU system이 2개 있을 경우  
  world_size = machine 개수(2) * machine 당 process 개수(4)  

- `torch.distributed.init_process_group()` :  
  - 분산 학습 환경 초기화 : 분산 학습하는 각 process 간의 통신을 설정  
  - backend : 'gloo' for CPU, 'nccl' for GPU, 'mpi' for 고성능  
  - init_method : 각 process가 서로 탐색하는 방법(url)  
  예시 : 'env://', f'tcp://127.0.0.1:11203'

- `torch.utils.data.distributed.DistributedSampler()` :  
  - world_size(총 process 개수)만큼 dataset을 분할하여 모든 process가 동일한 양의 dataset을 갖도록 함  
  - DistributedSampler는 각 epoch마다 dataset을 무작위로 분할

- `torch.utils.data.DataLoader()` :  
  - shuffle=False :  
  보통 training일 때는 일반화를 위해 shuffle=True로 두지만  
  분산 학습을 할 때는 같은 epoch 내에서  
  각 process가 서로 다른 dataset을 처리하기 위해 (중복 방지)  
  shuffle=False로 설정  
  - num_workers : cpu data load할 때 multi-processing core 개수  
  - pin_memory=True :  
  data load한 장치(CPU)에서 GPU로 data를 옮길 때  
  host memory가 아닌 CPU의 page-locked memory로 할당하고  
  GPU는 이를 참조하여 복사하므로 전송 시간을 단축  
  pin_memory=True와 non_blocking=True는 함께 사용  
  - drop_last=True : 나눠떨어지지 않는 마지막 batch를 버림

```Python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import timedelta

# 길이가 다른 input들을 batch로 묶기 위해 padding해주는 함수
# DataLoader()에서 사용
def _collate_fn(samples):
    # ...

# main_worker : 각 process가 실행하는 함수
def main_worker(process_id, args):

    rank = args.machine_id * args.num_processes + process_id
    
    world_size = args.num_machines * args.num_processes
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timedelta(300))
    
    ################################################################################################
    
    train_dataset = CustomDataset(dataset_path)
    # train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1302,), (0.3069,))]))  

    # machine 당 process 수로 나눔
    batch_size = int(args.batch_size / args.num_processes) 
    num_workers = int(args.num_workers / args.num_processes)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True, drop_last=True, sampler=train_sampler)
```

### Train

- `torch.multiprocessing.spawn()` :  
  - main_worker : 각 process가 실행하는 함수
  - nprocs : machine 당 process 개수인 4로 설정  
  main_worker()의 첫 번째 argument는 process_id인 0~3이 됨 
  - args : main_worker()에 추가로 전달할 tuple 형태의 argument  
  - join  
  - daemon  
  - start_method

- `main_worker()` : 각 process가 실행하는 함수  
1. `model` initialize, and set cuda, and parallelize  
  - nn.parallel.DistributedDataParallel :  
  각 model 복사본은 각자의 optimizer를 이용해 gradient를 구하고  
  rank=0의 process와 통신하여 gradient의 평균을 구해서 backpropagation 진행  
  GIL(global interpreter lock)의 제약을 해결 
2. `wandb` init  
  - wandb.init() : wandb 초기화  
  vars(args)는 args 객체의 __dict__ 속성을 반환  
  {'transforms' : 'BaseTransform', 'crop_size' : 224}과 같이 반환  
  - wandb.watch() : wandb 기록  
  모든 param.의 gradient를 기록  
  arg.log_interval-번째 batch마다 log 기록
3. `optimizer, scheduler` initialize  
4. load `checkpoint`  
5. `train` with `barrier`  
  - torch.distributed.barrier() :  
  분산 학습 환경에서  
  모든 process가 이 장벽에 도달할 때까지 대기하여  
  모든 process가 synchronize된 상태에서 훈련이 진행되도록 함  
  - torch.cuda.empty_cache() :  
  더 이상 사용하지 않는 tensor들을 GPU cached memory에서 해제  
  장점 : GPU memory 확보  
  단점 : 너무 자주 호출하면 메모리 할당/해제에 따른 성능 저하 발생
6. `wandb` and `distributed` finish  

```Python
from importlib import import_module
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from utils import *
from tqdm import tqdm
import wandb

def main():
    args = arg_parse()
    fix_seed(args.random_seed)
    
    # rank=0인 process를 실행하는 system의 IP 주소
    # rank=0인 system이 모든 backend 통신을 설정!
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # 해당 system에서 사용 가능한 PORT           
    os.environ['MASTER_PORT'] = '8892'

    mp.spawn(main_worker, nprocs=args.num_processes, args=(args,))

def main_worker(process_id, args):
    global best_acc
    best_acc = 0.0

    # 1. model initialize, and set cuda, and parallelize
    model = MyFMANet()

    torch.cuda.set_device(process_id)
    model.cuda(process_id)
    criterion = nn.NLLLoss().cuda(process_id) # criterion = nn.CrossEntropyLoss(reduction='mean').cuda(process_id)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[process_id])

    # 2. wandb init
    if rank == 0:
        wandb.init(project=args.prj_name, name=f"{args.exp_name}", entity="semyeongyu", config=vars(args))
        wandb.watch(model, log='all', log_freq=args.log_interval)

    # 3. optimizer, scheduler initialize
    optimizer = getattr(import_module("torch.optim"), args.optimizer)(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
    scheduler = getattr(import_module("torch.optim.lr_scheduler"), args.lr_scheduler)(optimizer, T_max=args.period, eta_min=0, last_epoch=-1, verbose=True)
    # T_max : 주기 1번 도는 데 걸리는 step 수 / eta_min : lr의 최솟값 / last_epoch : 학습 시작할 때의 epoch

    # 4. load checkpoint
    if args.resume_from:
        start_epoch, model, optimizer, scheduler = load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank)
    else:
        start_epoch = 0

    # 5. train with barrier
    model.train()

    dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch) # train_sampler가 epoch끼리 동일하게 data 분할하는 것을 방지하기 위해

        optimizer.zero_grad() # epoch마다 gradient 초기화

        train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        dist.barrier()

        if rank == 0:
            val_acc, val_loss = validate(val_loader, model, criterion, epoch, args)

            if (best_top1 < val_acc):
                best_top1 = val_acc # best_top1은 global var.
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'best_top1': best_top1,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, os.path.join(args.checkpoint_dir, args.exp_name), f"{epoch}_{round(best_top1, 4)}.pt"
                )
        
        torch.cuda.empty_cache() 
    
    # 6. wandb and distributed finish
    if rank == 0:
        wandb.run.finish()

    dist.destroy_process_group()
```

```Python
def train():
    for batch_i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        # pin_memory=True와 non_blocking=True는 함께 사용

        if batch_i % 10 == 0 and process_id == 0:
            # ...
```

```Python
def validate():
```

### Utils

- `argument parser`

```Python
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--transforms", type=str, default="BaseTransform")
    parser.add_argument("--crop_size", type=int, default=224)

    args = parser.parse_args()

    return args
```

- `seed`

```Python
import random
import torch
import numpy as np

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
```

- `checkpoint` : dictionary of elements below
  - epoch
  - model.state_dict()
  - best_acc
  - optimizer.state_dict()
  - scheduler.state_dict()

```Python
def save_checkpoint(checkpoint, saved_dir, file_name):
    os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, output_path) # checkpoint : dictionary

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank=-1):
    # checkpoint_path : ".../240325.pt"
    if rank != -1: # 분산학습 yes
        map_location = {"cuda:%d" % 0 : "cuda:%d" % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else: # 분산학습 no
        checkpoint = torch.load(checkpoint_path)

    start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return start_epoch, model, optimizer, scheduler
```

- `augmentation`

```Python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class BaseTransform(object):
    def __init__(self, crop_size = 224):
        self.transform = A.Compose(
            [   
                A.RandomResizedCrop(crop_size, crop_size),
                A.HorizontalFlip(),
                A.Normalize(),
                ToTensorV2() # albumentations에서는 normalize 이후에 ToTensorV2를 사용해줘야 함 (여기서 어차피 shape (C,H,W)로 변경)
            ]
        )

    def __call__(self, img):
        # BaseTransform()은 nn.Module을 상속한 게 아니므로 forward를 구현해도 __call__과 연결되어 있지 않음
        # 따라서 __call__()을 직접 구현해줘야 함
        return self.transform(image=img)
```

### Transformer

