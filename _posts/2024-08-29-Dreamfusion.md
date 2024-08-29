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

- `camera` :  
  - 3D model을 `bounded sphere` 내부로 제한하고,  
  spherical coordinate(구 표면)에서 camera 위치를 sampling하여  
  구의 원점을 바라보도록 camera 각도 설정  
  - width(64)에 0.7 ~ 1.35의 상수값을 곱하여 focal length 설정

- `light` :  
  - camera 위치를 중심으로 한 확률분포로부터 light의 위치를 sampling하고  
  (어떤 확률분포 `????`)  
  light 색상도 sampling

### NeRF Rendering with shading

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    albedo : NeRF가 예측한 color
</div>

- rendering 방법 :  
  1. albedo $$\rho$$ 만으로 rendering  
  (기존 NeRF와 동일)  
  2. albedo $$\rho$$ 뿐만 아니라 shading하여 rendering  
  3. albedo $$\rho$$ 를 white $$(1, 1, 1)$$ 로 바꾼 뒤 shading하여 rendering  

- `Shading`의 역할 :  
  - shading 없이 $$\rho$$ 만으로 rendering하면  
  평평한 3D model이 나와도 점수 높게 나옴  
  - shading으로 (빛 반사에 따른) shape 정보까지 고려해서 rendering하면  
  `volume 있는` 3D model이 되도록 촉구

- `NeRF MLP` $$\theta$$ :  
  - MLP output : volume density $$\tau$$ 와 albedo $$\rho$$

- `Normal` $$n$$ :  
$$n = - \frac{\nabla_{\mu} \tau}{\| \nabla_{\mu} \tau \|}$$  
where $$n$$ 은 물체 표면의 법선벡터
  - normal vector의 방향은  
  volume density $$\tau$$ 가 가장 급격하게 변하는 방향, 즉 $$\nabla_{\mu} \tau$$ 의  
  반대 방향

- `Shading` $$s$$ :  
$$s = (l_p \circ \text{max}(0, n \cdot \frac{l - \mu}{\| l - \mu \|})) + l_a $$  
where $$l_p$$ 는 light 좌표 $$l$$ 에서 나오는 light(광원) 색상  
where $$l_a$$ 는 ambient light(환경 조명) 색상  
where $$\mu$$ 는 shading 값을 계산할 surface 위 point 좌표  
where $$\circ$$ 는 element-wise multiplication
  - $$n \cdot (l - \mu)$$ 는 표면에서의 normal vector와 표면에서 광원까지의 vector 간의 내적이며,  
  이는 Lambertian(diffuse) reflectance(난반사)에 의해 광원의 빛이 반사되는 정도를 나타냄  
  왜냐하면, 빛이 표면에 수직으로 들어올수록 많이 반사됨  
  - 만약 빛이 표면 반대쪽에 있어서 내적 값  $$n \cdot (l - \mu)$$ 이 음수일 경우  
  난반사에 의해 광원의 빛이 반사되는 정도는 0  
  - $$l_p \circ \text{난반사 정도} + l_a$$ 에 의해  
  `광원`의 색상 $$l_p$$ 는 물체 `표면의 난반사 정도에 따라` 반영되고  
  `환경 조명`의 색상 $$l_a$$ 는 물체의 `모든 표면에 일정하게` 반영됨

- `Color` $$c$$ :  
$$c = \rho \circ s$$ 또는 $$c = \rho$$  


### Diffusion loss with conditioning

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- `Latent Diffusion` model :  
  - image $$x$$ 가 아니라 encoder를 거친 image latent vector $$z, \ldots z_T$$ 에 대해 noising, denoising 수행  
  - noisy $$z_T$$ 와 text embedding vector $$\tau_{\theta}$$ (conditioning)을 concat한 뒤  
  denoising하여 input image와 유사한 확률 분포를 갖도록 학습

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- TBD


### Optimization

TBD

## Rendering

### Structure

TBD

### Geometry Regularizer

TBD

## SDS Loss

TBD

## Pseudo Code

TBD

## Experiment

TBD

## Limitation

TBD

## Question

TBD