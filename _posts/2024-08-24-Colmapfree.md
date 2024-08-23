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

- colmap (SfM) library : 각 input image에 대한 camera pose 계산
단점 : 시간 많이 걸리고, feature 추출 오차에 대해 민감성이 있고, 반복적인 영역을 처리하는 데 어려움

- colmap-free 3D GS
initialization을 위해 Nope-Nerf 랑 비슷하게 depth map 사용!
한 번에 모든 프레임을 최적화하는 것이 아니라...
목표 : 주어진 frame t-1에 대해 local 3D Gaussian 집합을 구성하고, frame t로 3D Gaussian을 변환할 수 있는 "affine transformation을 학습하자"!

1. local 3DGS for relative pose estimation
1-1. initialization from a single view : 
monocular depth network 활용해서 depth map D_t 생성
intrinsic + monocular depth 이용해서 3DGS 초기화
초기화 후 렌더링 된 이미지 R(G_t) 와 현재 frame I_t 간의 photometric loss(L1 & D-SSIM)를 최소화하기 위해 3D Gaussian 집합 G_t를 학습
1-2. pose estimation by 3D Gaussian Transformation :
relative camera pose 추정을 위해 learnable SE-3 affine transformation T_t로 3D Gaussian 집합인 G_t를 frame t+1의 G_t+1로 변환
transform 후 렌더링 된 이미지 R(T_t \cdot G_t) 와 다음 frame I_t+1 간의 photometric loss를 최소화하기 위해 affine transformation T_t 학습

2. Global 3DGS with progressively growing
2-1. local 3DGS를 활용해서 frame t와 frame t+1 사이의 relative camera pose를 estimate
2-2. global 3DGS의 경우 estimated relative pose / observed 2 frames 를 입력으로 사용해서 N iter.에 걸쳐 모든 속성과 함께 3D Gaussian 집합을 업데이트

3. Experiment
GS가 아니라 pose가 주어지지 않는 Nerf 방법들과 비교했고, 더 나았음

+ chamfer distance : point cloud 집합인 P_i와 P_j가 서로 가까워지도록 하는 loss (SDS loss 쪽 공부해보면 흔히 등장)