---
layout: distill
title: Pytorch Basic Code (DDP)
date: 2024-07-08 11:00:00
description: Dataset, DataLoader, Train, Attention, ...
tags: pytorch
categories: others
thumbnail: assets/img/2024-07-08-PytorchBasic/1m.PNG
bibliography: 2024-07-08-PytorchBasic.bib
giscus_comments: false
disqus_comments: true
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

## Pytorch Basic Code (DistributedDataParallel ver.)

### Deal with json, image, csv

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/3m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

```Python
import os
import json
import pandas as pd
import csv
from PIL import Image
import cv2

# read json
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    f.close()

# read image 방법 1.
img = Image.open(image_path).convert('RGB') # PIL image object in range [0, 255]
img.show()

# read image 방법 2.
img = cv2.imread(image_path) # np.ndarray of shape (H, W, C) in range [0, 255] in BGR mode
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

### Convert to Tensor

data.py의 CustomDataset(torch.utils.data.Dataset)에서 image는 `shape (C, H, W) tensor`여야 하기 때문에  
PIL.Image.open() 또는 cv2.imread()로 얻은  
PIL image object 또는 np.ndarray를 적절한 shape 및 range의 tensor로 변환해주어야 한다  

- PIL image object $$\rightarrow$$ Tensor  
  - torchvision.transforms.ToTensor()
  - np.array(), torch.tensor()
  - getdata(), torch.tensor()  

- np.ndarray $$\rightarrow$$ Tensor
  - torch.tensor()

```Python
PIL_img = Image.open(image_path).convert('RGB') # PIL image object of size (W, H) in range [0, 255]

# 방법 1. torchvision.transforms.ToTensor()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor() # convert to tensor in range [0., 1.]
])
img = transform(PIL_img) # tensor of shape (C, H, W) in range [0., 1.]

# 방법 2. np.array(), torch.tensor()
img = np.array(PIL_img) # np.ndarray of shape (H, W, C) in range [0., 255.]
img = torch.tensor(img.transpose((2, 0, 1)).astype(float)).mul_(1.0) / 255.0 # tensor of shape (C, H, W) in range [0., 1.]

# 방법 3. getdata(), torch.tensor()
img_data = PIL_img.getdata()
img = torch.tensor(img_data, dtype=torch.float32) # tensor of shape (H*W*C,) in range [0, 255]
img = img.view(PIL_img.size[1], PIL_img.size[0], 3).permute(2, 0, 1) / 255.0 # tensor of shape (C, H, W) in range [0., 1.]
```

```Python
img = cv2.imread(image_path) # np.ndarray of shape (H, W, C) in range [0., 255.]

img = torch.tensor(img.transpose((2, 0, 1)).astype(float)).mul_(1.0) / 255.0 # tensor of shape (C, H, W) in range [0., 1.]
```

### Create Dataset

- `data augmentation` :  
  - Resize
  - ToTensor
  - RandomHorizontalFlip
  - RandomVerticalFlip
  - Normalize
  - RandomRotation
  - RandomAffine  
    - shear
    - scale (zoom-in/out)
  - RandomResizedCrop
  - ColorJitter
  - GaussianBlur

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/4m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


```Python
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
import random

# Create Dataset
class CustomDataset(Dataset):
    def __init__(self, args, mode):
        # lazy-loading :
        # load할 data가 너무 크다면 __init__()에서는 load할 파일명만 저장해놓고 __getitem__()에서 필요할 때마다 load
        self.args = args
        self.mode = mode
        
        if mode == 'train':
            self.data_path = os.path.join(args.data_path, 'train_blur')
        elif mode == 'val':
            self.data_path = os.path.join(args.data_path, 'val_blur')
        elif mode == 'test':
            self.data_path = os.path.join(args.data_path, 'test_blur')
        
        # a list of data/train_blur/*m.PNG
        self.blur_path_list = sorted(glob.glob(os.path.join(self.data_path, '*m.PNG')))
        
        # a list of data/train_sharp/*m.PNG
        self.sharp_path_list = [os.path.normpath(path.replace('blur', 'sharp') for path in self.blur_path_list)]

    def __getitem__(self, idx):
        # should return float tensor!!
        blur_path = self.blur_path_list[idx]
        # np.ndarray of shape (H, W, C) in range [0, 255]
        blur_img = cv2.imread(blur_path) 

        if self.mode == 'train':
            sharp_path = self.sharp_path_list[idx]
            sharp_img = cv2.imread(sharp_path)
            
            # np.ndarray of shape (pat, pat, C) where pat is patch_size
            blur_img, sharp_img = self.augment(self.get_random_patch(blur_img, sharp_img)) 
            
            # tensor of shape (C, pat, pat) in range [0, 1]
            return self.np2tensor(blur_img), self.np2tensor(sharp_img) 
        
        elif self.mode == 'val':
            sharp_path = self.sharp_path_list[idx]
            sharp_img = cv2.imread(sharp_path)
            return self.np2tensor(blur_img), self.np2tensor(sharp_img)
        
        elif self.mode == 'test':
            return self.np2tensor(blur_img), blur_path

    def np2tensor(self, x):
        # input : shape (H, W, C) / range [0, 255]
        # output : shape (C, H, W) / range [0, 1]
        ts = (2, 0, 1)
        x = torch.tensor(x.transpose(ts).astype(float)).mul_(1.0) # _ : in-place
        x = x / 255.0 # normalize
        return x

    def get_random_patch(self, blur_img, sharp_img):
        H, W, C = blur_img.shape # shape (H, W, C)

        pat = self.args.patch_size # pat : patch size
        iw = random.randrange(0, W - pat + 1) # iw : range [0, W - pat]
        ih = random.randrange(0, H - pat + 1) # ih : range [0, H - pat]

        blur_img = blur_img[ih:ih + pat, iw:iw + pat, :] # shape (pat, pat, C)
        sharp_img = sharp_img[ih:ih + pat, iw:iw + pat, :]

        return blur_img, sharp_img # shape (pat, pat, C)

    def augment(self, blur_img, sharp_img):
        # random horizontal flip
        if random.random() < 0.5:
            blur_img = blur_img[:, ::-1, :] # Width-axis를 flip
            sharp_img = sharp_img[:, ::-1, :]
            '''
            flow-mask pair의 경우 C-dim.이 3 = 2(optical flow x, y) + 1(occlusion mask) 이므로
            shape (T, H, W, 3)의 flow-mask pair를 horizontal flip을 하려면
            flow = flow[:, :, ::-1, :]
            flow[:, :, :, 0] *= -1
            '''
            
        # random vertical flip
        if random.random() < 0.5:
            blur_img = blur_img[::-1, :, :] # Height-axis를 flip
            sharp_img = sharp_img[::-1, :, :]
            '''
            flow = flow[:, ::-1, :, :]
            flow[:, :, :, 1] *= -1
            '''

        return blur_img, sharp_img

    def __len__(self):
        return len(self.path_list)
```

### DataLoader

- DataParallel(DP) vs DistributedDataParallel(DDP) :  
[Pytorch DP and DDP](https://tkayyoo.tistory.com/27#tktag2)  
  - DataParallel(DP) :  
  single-process  
  multi-thread  
  single-machine  
  - DistributedDataParallel(DDP) :  
  multi-process  
  single-machine과 multi-machine 모두 가능  

- `rank` :  
  - 전체 distributed system에서 process 순서  
  - 4-GPU system이 2개 있을 경우  
  rank = machine 번호(0~1) * machine 당 process 개수(4) + process 번호(0~3)  
    - rank :  
    `rank = int(os.environ['RANK'])`  
    torch.distributed.init_process_group() 한 이후에  
    rank = torch.distributed.get_rank()
    - machine 당 process 개수 :  
    `num_gpu = torch.cuda.device_count()`  
    - process id (local rank) :  
    process_id = rank % num_gpu  
    torch.cuda.set_device(process_id)  
    model.cuda(process_id)  
    `process_id = torch.cuda.current_device()`  
    - world_size :  
    torch.distributed.init_process_group() 한 이후에  
    world_size = torch.distributed.get_world_size()
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

- `_collate_fn(input)` :  
  - DataLoader()에서 1개의 batch로 묶을 때 사용하는 custom 전처리 함수  
  - input :  
    - 1개의 batch에 해당하는 입력  
    - Dataset(torch.utils.data.Dataset)의 __getitem__(self, idx)이 return img, target 형태일 때  
    [(img1, target1), (img2, target2), ...]의 형태  
  - output :  
    - for iter, (x, y) in enumerate(dataloader): 의 x, y  
  - 예 : 길이가 다른 input들을 batch로 묶기 위해 padding, tokenization  
  img의 경우 CustomDataset()에서 augmentation으로 shape (C, H, W)로 통일해줬다면 (N, C, H, W)로 묶을 수 있지만,  
  object detection task에서 target의 경우 n_box가 image마다 다르므로 N batch로 묶기 위해 padding해주어야 함

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/5m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


```Python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import timedelta

def _collate_fn(samples):
    # ...

# main_worker : 각 process가 실행하는 함수
def main_worker(process_id, args):
    
    rank = args.machine_id * args.num_processes + process_id
    
    world_size = args.num_machines * args.num_processes
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timedelta(300))
    
    ###################################################################
    
    # for epoch in range ... 밖에서
    train_dataset = CustomDataset(args, 'train')
    '''
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose(
        [
        transforms.ToTensor(), 
        transforms.Normalize((0.1302,), (0.3069,))
        ]))  
    '''

    # machine 당 process 수로 나눔
    batch_size = int(args.batch_size / args.num_processes) 
    num_workers = int(args.num_workers / args.num_processes)

    # for epoch in range ... 안에서
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

1. `model` initialize, and set cuda, and parallelize  
  - nn.parallel.DistributedDataParallel :  
  각 model 복사본은 각자의 optimizer를 이용해 gradient를 구하고  
  rank=0의 process와 통신하여 gradient의 평균을 구해서 backpropagation 진행  
  GIL(global interpreter lock)의 제약을 해결  
2. `wandb` init  
  - wandb.init() : wandb 초기화  
    - config=vars(args) : vars(args)는 args 객체의 __dict__ 속성을 반환  
    {'transforms' : 'BaseTransform', 'crop_size' : 224}과 같이 반환  
  - wandb.watch() : wandb 설정  
    - log='all' : 모든 param.의 gradient를 기록  
    - log_freq : arg.log_interval-번째 batch마다 log 기록
  - wandb.log() : wandb 기록  
    - wandb.log(__dict__)  
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
  - train :  
  args.accumulation_steps만큼 loss를 누적한 뒤 backward  
  args.accumulation_steps마다 gradient 및 measurement 초기화, rank=0 logging  
  - validation :  
  with torch.no_grad(): 로 gradient 누적 안 함!  
  또는  
  with torch.set_grad_enabled(false): 로 gradient 누적 안 함!
  - model.state_dict() :  
  torch.save() 또는 wandb.log()로 model.state_dict()를 별도 파일에 저장 가능  
  model_param = copy.deepcopy(model.state_dict())로 model.state_dict()를 코드 내 변수에 저장 가능
6. `wandb` and `distributed` finish  



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/6m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


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
    # DDP가 아니라면, main.py에 def main_worker()의 내용을 넣고, train.py에 class Runner 만들자
    '''
    class Runner:
        def __init__(self, args, model):
            self.args = args
            self.model = model
            pass
        def train(self, dataloader, epoch):
            pass
        def validate(self, dataloader, epoch):
            pass
        def test(self, dataloader):
            pass
    '''

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
    # T_max : 주기 1번 도는 데 걸리는 최대 iter. 수 / eta_min : lr의 최솟값 / last_epoch : 학습 시작할 때의 epoch

    # 4. load checkpoint
    if args.resume_from:
        start_epoch, model, optimizer, scheduler = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler, rank)
    else:
        start_epoch = 0

    # 5. train with barrier
    dist.barrier()

    for epoch in range(start_epoch, args.n_epochs):
        train_sampler.set_epoch(epoch) # train_sampler가 epoch끼리 동일하게 data 분할하는 것을 방지하기 위해

        optimizer.zero_grad() # epoch마다 gradient 초기화

        train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        dist.barrier()

        if rank == 0:
            val_acc, val_loss = validate(val_loader, model, criterion, epoch, args)
            
            # best acc일 때마다 save checkpoint
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
        wandb.finish()

    dist.destroy_process_group()
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/7m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


```Python
def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    model.train()
    train_acc, train_loss = AverageMeter(), AverageMeter() 
    # measurement of acc and loss

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (x, y_gt) in pbar:
        x, y_gt = x.cuda(non_blocking=True), y_gt.cuda(non_blocking=True) 
        # cuda device에 올려야 함
        # pin_memory=True와 non_blocking=True는 함께 사용

        # forward
        y_pred = model(x)

        # loss divided by accumulation_steps
        loss = criterion(y_pred, y_gt) / args.accumulation_steps

        # gradient 누적
        loss.backward()
        
        # measurement
        train_acc.update(
            topk_accuracy(y_pred.clone().detach(), y_gt).item(), x.size(0))
        train_loss.update(loss.item() * args.accumulation_steps, x.size(0))

        # args.accumulation_steps만큼 loss를 누적한 뒤 평균값으로 backward
        if (step+1) % args.accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

            # backward
            optimizer.step()
            scheduler.step()

            # gradient 초기화
            optimizer.zero_grad()

            # logging
            dist.barrier()
            if rank == 0:
                # wandb log
                wandb.log(
                    {
                        "Training Loss": round(train_loss.avg, 4),
                        "Training Accuracy": round(train_acc.avg, 4),
                        "Learning Rate": optimizer.param_groups[0]['lr']
                    }
                )

                # tqdm log
                description = f'Epoch: {epoch+1}/{args.n_epochs} || Step: {(step+1)//args.accumulation_steps}/{len(train_loader)//args.accumulation_steps} || Training Loss: {round(train_loss.avg, 4)}'
                pbar.set_description(description)

                # measurement 초기화
                train_loss.init()
                train_acc.init()
    
    return train_loss.avg
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/8m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


```Python
def validate(val_loader, model, criterion, epoch, args):
    model.eval()
    val_acc, val_loss = AverageMeter(), AverageMeter() # measurement of acc and loss

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    with torch.no_grad(): # validation은 gradient 누적 안 함!!
        for step, (x, y_gt) in pbar:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            
            # forward
            y_pred = model(x)
            loss = criterion(y_pred, y_gt)
            
            # measurement
            val_acc.update(topk_accuracy(y_pred.clone().detach(), y_gt).item(), x.size(0)) 
            val_loss.update(loss.item(), x.size(0))

            # tqdm log
            description = f'Epoch: {epoch+1}/{args.n_epochs} || Step: {step+1}/{len(val_loader)} || Validation Loss: {round(loss.item(), 4)} || Validation Accuracy: {round(val_acc.avg, 4)}'
            pbar.set_description(description)

    # wandb log
    wandb.log(
        {
            'Validation Loss': round(val_loss.avg, 4),
            'Validation Accuracy': round(val_acc.avg, 4)
        }
    )

    return val_acc.avg, val_loss.avg
```

### Utils

- `argument parser`

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/9m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


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

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/10m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


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

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/11m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


```Python
def save_checkpoint(checkpoint, saved_dir, file_name):
    os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, output_path) 
    # checkpoint : dictionary

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

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/12m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

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
                ToTensorV2() 
# albumentations에서는 normalize 이후에 ToTensorV2를 사용해줘야 함 (여기서 어차피 shape (C,H,W)로 변경)
            ]
        )

    def __call__(self, img):
# BaseTransform()은 nn.Module을 상속한 게 아니므로 forward를 구현해도 __call__과 연결되어 있지 않음
# 따라서 __call__()을 직접 구현해줘야 함
        return self.transform(image=img)
```

- `measurement`

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/13m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

```Python
class AverageMeter(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def topk_accuracy(pred, gt, k=1):
    # pred : shape (N, class)
    # gt : shape (N,)
    _, pred_topk = pred.topk(k, dim=1)
    n_correct = torch.sum(pred_topk.squeeze() == gt)

    return n_correct / len(gt)
```

### Multi-Attention

- FMA-Net (2024) <d-cite key="FMANet">[1]</d-cite>의 Multi-Attention 구현  
출처 : [FMA-Net Code](https://github.com/KAIST-VICLab/FMA-Net)  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/2m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

> Multi-Attention :  
- `CO(center-oriented)` attention :  
better align $$\tilde F_{w}^{i}$$ to $$F_{c}^{0}$$ (center feature map of initial temporally-anchored feature)  
- `DA(degradation-aware)` attention :  
$$\tilde F_{w}^{i}$$ becomes globally adaptive to spatio-temporally variant degradation by using degradation kernels $$K^{D}$$  

- CO attention :  
$$Q=W_{q} F_{c}^{0}$$  
$$K=W_{k} \tilde F_{w}^{i}$$  
$$V=W_{v} \tilde F_{w}^{i}$$  
$$COAttn(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d}})V$$  
실험 결과, $$\tilde F_{w}^{i}$$가 자기 자신(self-attention)이 아니라 $$F_{c}^{0}$$과의 relation에 집중할 때 better performance  

- DA attention :  
CO attention과 비슷하지만,  
Query 만들 때 $$F_{c}^{0}$$ 대신 $$k^{D, i}$$ 사용  
$$\tilde F_{w}^{i}$$ becomes globally adaptive to spatio-temporally-variant degradation  
$$k^{D, i} \in R^{H \times W \times C}$$ : degradation features adjusted by conv. with $$K^{D}$$ (motion-aware spatio-temporally-variant degradation kernels) 에 대해  
$$Q=W_{q} k^{D, i}$$  
DA attention은 $$Net^{D}$$ 말고 $$Net^{R}$$ 에서만 사용  

- `nn.Conv2d()` :  
  - $$H_{out} = \lfloor 1 + \frac{H_{in} + 2 \times pad - dilation \times (K-1) - 1}{stride} \rfloor$$  
  - argument :  
    - groups :  
    shape ($$C_{in}$$, $$C_{out}$$, K, K) 대신 $$C_{in}$$, $$C_{out}$$ 을 groups-개로 쪼개서  
    shape ($$\frac{C_{in}}{groups}$$, $$\frac{C_{out}}{groups}$$, K, K)를 groups-번 실행하여 concat  
  - variable :  
    - weight : shape ($$C_{out}$$, $$\frac{C_{in}}{groups}$$, K, K)  
    - bias : shape ($$C_{out}$$,)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-08-PytorchBasic/14m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


```Python
import torch
import torch.nn as nn

class Attention(nn.Module):
    # Restormer (CVPR 2022) transposed-attention block
    # original source code: https://github.com/swz30/Restormer
    def __init__(self, dim, n_head, bias):
        super(Attention, self).__init__()
        self.n_head = n_head # multi-head for channel dim.
        self.temperature = nn.Parameter(torch.ones(n_head, 1, 1)) 
        # multi-head 별로 scale factor를 parameterize

        # W_q
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # W_kv
        self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        # W_o
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, f):
        # first input x : shape (N, C, H, W) -> makes key and value
        # second input f : shape (N, C, H, W) -> makes query
        N, C, H, W = x.shape

        # Apply W_q and W_kv
        q = self.q_dwconv(self.q(f)) # query q : shape (N, C, H, W)
        kv = self.kv_dwconv(self.kv_conv(x)) # kv : shape (N, 2*C, H, W)
        k, v = kv.chunk(2, dim=1) # key k and value v : shape (N, C, H, W)

        # Multi-Head Attention
        q = einops.rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        # query q : shape (N, C, H, W) -> shape (N, M, C/M, H * W)
        k = einops.rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        # key k : shape (N, C, H, W) -> shape (N, M, C/M, H * W)
        v = einops.rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        # value v : shape (N, C, H, W) -> shape (N, M, C/M, H * W)

        # matrix mul.을 할 spatial dim.을 normalize
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        '''
        - q @ k.transpose(-2, -1) = similarity :
          shape (N, M, C/M, C/M)
        - self.temperature = scale factor for each head :
          shape (M, 1, 1) -> shape (N, M, C/M, C/M) 
        '''
        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1) # convert to probability distribution

        out = (attn @ v) # shape (N, M, C/M, H*W)
        
        # Multi-Head Attention - concatenation
        out = einops.rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.n_head, h=H, w=W) 
        # shape (N, C, H, W)

        # Apply W_o
        out = self.project_out(out) # shape (N, C, H, W)

        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        
        # learnable param.
        self.weight = nn.Parameter(torch.ones(normalized_shape)) # shape (C,)
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) # shape (C,)
    
    def forward(self, x):
        # x : shape (N, C, H, W)
        # LayerNorm : dim. C에 대해 normalize
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class MultiAttentionBlock(nn.Module):
    def __init__(self, dim, n_head, ffn_expansion_factor, bias, is_DA):
        super(MultiAttentionBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        # center-oriented attention
        self.co_attn = Attention(dim, n_head, bias) 
        self.norm2 = LayerNorm(dim)
        self.ffn1 = FeedForward(dim, bias)

        if is_DA:
            self.norm3 = LayerNorm(dim)
            # degradation-aware attention
            self.da_attn = Attention(dim, n_head, bias) 
            self.norm4 = LayerNorm(dim)
            self.ffn2 = FeedForward(dim, bias)

    def forward(self, Fw, F0_c, Kd):
        Fw = Fw + self.co_attn(self.norm1(Fw), F0_c)
        Fw = Fw + self.ffn1(self.norm2(Fw))

        if Kd is not None:
            Fw = Fw + self.da_attn(self.norm3(Fw), Kd)
            Fw = Fw + self.ffn2(self.norm4(Fw))

        return Fw
```

### nn module

- layer :  
  - nn.Conv2d
  - nn.RNNCell
  - nn.LSTMCell
  - nn.GRUCell
  - nn.Transformer

- activation :  
  - nn.Sigmoid
  - nn.ReLU
  - nn.LeakyReLU
  - nn.Tanh
  - nn.Softplus

- augograd :  
  - update해야 하는 tensor (model param.)는 requirs_grad=True로 설정하자!  
  - y.retain_grad() : leaf node가 아닌 tensor의 gradient는 계산 후 날라가는데  
  y.regain_grad()를 하면 y.grad가 사라지지 않도록 붙잡아둠