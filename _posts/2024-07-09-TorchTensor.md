---
layout: distill
title: Pytorch Tensor
date: 2024-07-09 15:00:00
description: import torch
tags: pytorch
categories: cv-tasks
thumbnail: assets/img/2024-07-09-TorchTensor/1.png
giscus_comments: false
disqus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
# featured: true
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

## Pytorch Tensor Summary

### Python 문법

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Create and Access Tensor

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Tensor slice indexing

- a[start, end, step]에서 start, end는 시작 및 끝 위치만 정하고, 방향은 step이 정함  
e.g. a = np.arange(10)[8:5] 는 []  
e.g. a = np.arange(10)[8:5:-1] 는 [8, 7, 6]  

- a[1,:,:,:,:,2] 대신 a[1,...,2] 으로 생략 가능

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Tensor list(tensor) & boolean indexing

- Advanced Indexing :  
tensor a : shape (5, 5, 3)  
tensor b : shape (3,)  
tensor c : shape (3,)  
tensor d : shape (2,)  
a[b, d, :] : shape (3, 2, 3)  
a[b, c, :] : shape (3, 3)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/6.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/7.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### gather, scatter_add indexing

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/8.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### where condition

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/9.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Timing

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/10.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Reshape, Permute

- dimension 추가 :  
  - 방법 1. a.unsqueeze(1)
  - 방법 2. a[:, None]

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/11.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### element-wise, reduction operation, concatenation

- np에선 np.concatenate([a, b], axis=0)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/12.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Matrix operation

- a : shape (4, 3, 2), b : shape (4, 2, 3)일 때  
np.dot(a, b) : shape (4, 3, 4, 3) 이고  
a@b : shape (4, 3, 3) 이므로  
batched matrix multiplication 수행하는 a@b 권장

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/13.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/14.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Broadcasting

- Broadcasting 규칙 :  
  - 두 tensor의 차원이 다를 경우, 더 작은 차원의 tensor의 shape 앞에 1이 추가되어 동일한 차원이 된다
  - 두 텐서의 각 차원에서 크기가 같거나, 한쪽의 크기가 1인 경우에만 broadcasting 가능

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/15.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### In-place, GPU

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/16.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

