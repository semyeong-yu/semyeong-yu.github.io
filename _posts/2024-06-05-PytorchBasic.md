---
layout: distill
title: Pytorch Basic Code
date: 2024-06-05 17:00:00
description: Dataset, DataLoader, Train, ...
tags: pytorch
categories: cv-tasks
# thumbnail: assets/img/2024-06-05-PytorchBasic/1.png
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
  - rank = 0인 process에 대해서만 train log 출력

- `world_size` :  
  - 총 process 개수  
  - 4-GPU system이 2개 있을 경우  
  world_size = machine 개수(2) * machine 당 process 개수(4)  

- torch.distributed.init_process_group() :  
분산 학습을 하는 각 process 간의 통신을 위해 사용

- torch.utils.data.distributed.DistributedSampler() :  
  - world_size(총 process 개수)만큼 dataset을 분할하여 모든 process가 동일한 양의 dataset을 갖도록 함  
  - DistributedSampler는 각 epoch마다 dataset을 무작위로 분할

- torch.utils.data.DataLoader() :  
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

# 길이가 다른 input들을 batch로 묶기 위해
# padding해주는 함수
def _collate_fn(samples):
    # ...

# main_worker : 각 process가 실행하는 함수
def main_worker(process_id, args):

    rank = args.machine_id * args.num_processes + process_id
    
    world_size = args.num_machines * args.num_processes
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # backend : 'gloo' for CPU, 'nccl' for GPU, 'mpi' for 고성능
    # init_method : 각 process가 서로 탐색하는 방법(url)
    ################################################################################################
    train_dataset = CustomDataset(dataset_path)
    # train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1302,), (0.3069,))]))  

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=_collate_fn, pin_memory=True, drop_last=True, sampler=train_sampler)
```

### Train

1. `initialize`  
2. `load checkpoint`
3. `set cuda process_id`
4. `parallelize model`  
  - 각 model 복사본은 각자의 optimizer를 이용해 gradient를 구하고  
  rank=0의 process와 통신하여 gradient의 평균을 구해서 backpropagation 진행  
  - GIL(global interpreter lock)의 제약을 해결
5. `train`

```Python
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
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
    # main_worker : 각 process가 실행하는 함수
    # nprocs : machine 당 process 개수

def main_worker(process_id, args):
    # 1. initialize
    model = MyFMANet()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    scheduler = # ...

    # 2. load checkpoint
    start_epoch, model, optimizer, scheduler = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    # 3. set cuda process_id
    torch.cuda.set_device(process_id)
    model.cuda(process_id)
    criterion = nn.NLLLoss().cuda(process_id)

    # 4. parallelize model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[process_id])
    
    # 5. train
    model.train()

    for epoch in range(start_epoch, args.epochs):
        for batch_i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # pin_memory=True와 non_blocking=True는 함께 사용

            if batch_i % 10 == 0 and process_id == 0:
                # ...
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
    if rank != -1: # distributed
        map_location = {"cuda:%d" % 0 : "cuda:%d" % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_path)

    start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return start_epoch, model, optimizer, scheduler
```

### Transformer

