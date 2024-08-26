---
layout: distill
title: COLMAP-Free 3D Gaussian Splatting
date: 2024-08-24 11:00:00
description: COLMAP-Free 3D Gaussian Splatting (CVPR 2024)
tags: COLMAP SfM GS depth pose rendering 3d
categories: 3d-view-synthesis
thumbnail: assets/img/2024-08-24-Colmapfree/1.png
giscus_comments: false
disqus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
# featured: true
toc:
  - name: Introduction
  - name: Overview
  - name: Local 3DGS for Relative Pose Estimation
  - name: Global 3DGS with Progressively Growing
  - name: Experiment
  - name: Limitation
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

## COLMAP-Free 3D Gaussian Splatting

#### Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, Xiaolong Wang

> paper :  
[https://arxiv.org/abs/2312.07504](https://arxiv.org/abs/2312.07504)  
project website :  
[https://oasisyang.github.io/colmap-free-3dgs/](https://oasisyang.github.io/colmap-free-3dgs/)  
pytorch code :  
[https://github.com/NVlabs/CF-3DGS](https://github.com/NVlabs/CF-3DGS)  

## Introduction

- 기존 novel-view-synthesis :  
  - input images  
  $$\rightarrow$$ COLMAP library for SfM `pcd,  camera pose` 계산  
  $$\rightarrow$$ NeRF or 3DGS  
  - 단점 : 시간 많이 걸리고, feature 추출 오차에 대해 민감성이 있고, 반복적인 영역을 처리하는 데 어려움

- Motivation :  
`pose estimation`과 `novel-view-synthesis`를  
COLMAP과 3DGS로 나눠서 하지 말고  
`end-to-end로 동시에` 할 수는 없을까?

- Related Work :  
사전에 COLMAP library 사용하지 않기 위해  
`BARF, Nope-NeRF, L2G-NeRF` 등  
여러 방법들이 제안되어 왔지만  
여러 한계 있음  
  - perturbation이 적어야 함
  - `camera motion의 범위`가 너무 넓으면 안 됨  
  (Nope-NeRF 등은 pose를 직접 optimize하는 게 아니라 ray casting process를 optimize하는 간접적인 방법이라서 camera 이동이 커지면 optimize 난이도가 복잡해짐)
  - `training time`이 너무 오래 걸림
  - NeRF-based 기법들은 MLP-based implicit method이므로  
  3DGS처럼 explicit pcd를 요구하는 method에 적용하기 어렵
  - regularization term이 많아져서 복잡하고 geometric prior를 요구하기도 함

- COMALP-Free 3D GS :  
  - 3DGS가 `explicit` representation (pcd 등) 을 활용할 수 있기 때문에 새로운 접근이 가능해졌다  
    - temporal continuity data (video sequence)와  
    explicit representation data (3DGS)를 이용해서  
    pose estimation과 novel view synthesis를 동시에 수행  
  - Local 3DGS :  
    - `initialization` 위해 Nope-Nerf 랑 비슷하게 monocular `depth-map` 사용
    - 목표 :  
    주어진 frame $$t-1$$ 에서의 local 3D Gaussian 집합을 구성하고,  
    frame $$t$$ 에서의 local 3D Gaussian 집합으로 변환할 수 있는  
    `relative camera pose (affine transformation)` 학습 
  - Global 3DGS :  
    - 목표 :  
    Local 3DGS에서 구한 relative camera pose를 기반으로  
    Global 3DGS를 순차적으로 점진적으로 계속 업데이트해서  
    entire scene `reconstruction` 결과가 깔끔하게 나타나도록 하자

- COLMAP vs 본 논문 :  
  - COLMAP : 100장의 images를 `한 번에` 넣고 camera pose를 optimize
  - 본 논문 : video sequence를 `순차적으로` 실시간으로 받으며 점진적으로 optimize

## Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-24-Colmapfree/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## Local 3DGS for Relative Pose Estimation

- Initialization from a Single View :  
  - initial frame을 monocular depth network (DPT)에 넣어 depth map $$D_1$$ 생성
  - 3D mean :  
  `initial frame` (2D 정보)과  
  `initial depth map` $$D_1$$ (3D 정보)와  
  `intrinsic` param. 이용해서  
  3D pcd로 투영하고, 이를 initial 3DGS mean point로 사용
  - opacity, SH-coeff., covariance(rotation, scale) :  
  L1, D-SSIM photometric loss로 `optimal (initial) Local 3DGS`를 5초 정도만에 구함  
  initial frame $$t = 1$$ 에 대해  
  $$G_t^{\ast} = \text{argmin}_{\alpha_{t}, c_t, r_t, s_t} L_{rgb}(R(G_t), I_t) = (1 - \lambda) L_1 + \lambda L_{D-SSIM}$$  

- Pose Estimation by 3D Gaussian Transformation :  
  - Gaussian 집합 $$G_t$$ 를 $$G_{t+1}$$ 로 올바르게 변환할 수 있는 learnable SE-3 affine transformation $$T_t$$ 를 찾아야 함  
  - 전제 : video로 찍은 연속적인 frame이므로 $$T_t$$ 의 값이 크지 않음
  - photometric loss로 `optimal relative camera pose(affine transformation)`을 10초 안에 구함  
  $$T_t^{\ast} = \text{argmin}_{T_t} L_{rbg} (R(T_t \odot G_t), I_{t+1})$$  
  where $$G_t$$ is `freezed` (self-rotation 등 방지)  
  (geometric transformation(camera movement)에만 집중)

## Global 3DGS with Progressively Growing

- Local 3DGS를 통해 optimal relative camera pose를 구했다  
  - 한계 : frame $$F$$ 와 frame $$F+t$$ 간의 relative camera pose를 단순히 $$\prod_{k=F}^{F+t} T_k$$ 처럼 곱으로 두면 오차가 점점 커져서  
  entire `scene reconstruction 결과가 noisy`  

- Global 3DGS :  
  - frame이 들어올 때마다 relative camera pose $$T_t$$ 와 frame $$t, t+1$$ 이용해서 `optimal Global 3DGS` 업데이트 (progressively growing)  
  - 어떻게 업데이트? :  
  frame $$t+1$$ 에는 frame $$t$$ 에서 `보지 못한 일부 영역` 들이 있으므로  
  새로운 frame에 대한 `under-reconstruction densification`에 초점을 두어  
  last frame까지 계속해서 점진적으로 scene reconstruction 수행  
  (last frame까지 계속 under-reconstruction 상황(보지 못했던 영역)이 발생할 것이라는 전제)  
  (새로운 테크닉은 아니고 3DGS에서의 adaptive density control과 동일)  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-24-Colmapfree/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Local 3DGS와 Global 3DGS는 iteratively optimized

## Experiment

- GS 말고 pose-free NeRF methods와 비교했을 때  
pose trajectory와 scene reconstruction 측면에서  
본 논문이 훨씬 더 좋은 성능  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-24-Colmapfree/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-24-Colmapfree/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-24-Colmapfree/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- COLMAP + 3DGS와 비교했을 때  
본 논문과 동일한 성능

- 우리는 pose estimation을 할 때 photometric loss에만 의존했음  
  - photometric loss에만 의존해서 relative camera pose를 구했기 때문에 $$RPE_r, RPE_t$$ 값이 Nope-NeRF보다 조금 높게 나타날 수 있음  
  - Nope-NeRF에서는 chamfer distance(point cloud 집합인 $$P_i$$ 와 $$P_j$$ 가 서로 가까워지도록 하는 point cloud loss) 추가하여 pose accuracy 높임  
    
- Nope-NeRF에서와 달리 본 논문에서 depth loss를 쓰면 pose accuracy는 비슷하고 novel view synthesis performance는 오히려 떨어지므로 depth loss는 안 씀

## Limitation

- camera pose와 3DGS를 동시에 순차적으로 optimize하므로  
video stream 혹은 ordered image 집합에만 적용 가능  
$$\rightarrow$$ unordered image 집합에도 적용하는 future work 필요