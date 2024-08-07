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

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/11.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### element-wise, reduction operation, concatenation

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-09-TorchTensor/12.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Matrix operation

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

