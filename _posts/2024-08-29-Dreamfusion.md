---
layout: distill
title: DreamFusion
date: 2024-08-29 11:00:00
description: Text-to-3D using 2D Diffusion (ICLR 2023)
tags: sds diffusion nerf 3d rendering 
categories: generative
thumbnail: assets/img/2024-08-29-Dreamfusion/1.png
giscus_comments: false
disqus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
# featured: true
toc:
  - name: Contribution
  - name: Overview
    subsections:
      - name: Random camera, light sampling
      - name: NeRF Rendering with shading
      - name: Diffusion loss with conditioning
      - name: Optimization
  - name: Rendering
    subsections:
      - name: Structure
      - name: Geometry Regularizer
  - name: SDS Loss
  - name: Pseudo Code
  - name: Experiment
  - name: Limitation
  - name: Question
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

## DreamFusion: Text-to-3D using 2D Diffusion (ICLR 2023)

#### Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

> paper :  
[https://arxiv.org/abs/2209.14988](https://arxiv.org/abs/2209.14988)  
project website :  
[https://dreamfusion3d.github.io/](https://dreamfusion3d.github.io/)  
pytorch code :  
[https://github.com/ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)  

## Contribution

- `SDS(Score Distillation) Loss` 처음 제시
- NeRF가 Diffusion(Imagen) model이 내놓을 만한 그럴 듯한 image를 합성하도록 함

## Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Overview  
  - initialize NeRF with random weight
  - for each iter.  
    - camera 위치와 각도, light 위치와 색상을 randomly sampling  
    $$P(camera), P(light)$$    
    - NeRF로 image rendering
    - NeRF param. $$\theta$$ 와 text embedding $$\tau$$ 이용해서 SDS loss 계산
    - update NeRF weight

### Random camera, light sampling

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- TBD

### NeRF Rendering with shading

albedo : nerf가 예측한 color

다양한 rendering 방법
1 평평한 3d model이 나와도 점수 높에 나옴
2 shading으로 shape 정보까지 고려해서 volume 있는 3d model이 되도록 촉구

### Diffusion loss with conditioning

latent diffusion model : noisy image latent vector $$Z_T$$ 와 text embedding vector $$\tau_{\theta}$$ (conditioning)을 concat한 뒤 denoising하여 input image와 유사한 확률 분포를 갖도록 학습

### Optimization



## Rendering

### Structure

### Geometry Regularizer

## SDS Loss

## Pseudo Code

## Experiment

## Limitation

## Question