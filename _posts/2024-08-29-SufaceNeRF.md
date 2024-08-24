---
layout: distill
title: NeRF-based 3D Surface Reconstruction
date: 2024-08-26 11:00:00
description: Key-Point Summary
tags: nerf rendering surface
categories: 3d-view-synthesis
thumbnail: assets/img/2024-08-29-SurfaceNeRF/1.png
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

## NeRF-based 3D Surface Reconstruction

## DVR

### TBD

- MLP와 함께 사용할 수 있도록 Differentiable Volumetric Rendering 방법을 제시

## IDR

- 직접 말고 implicitly
- BRDF : 빛이 물체에 튕겨서 우리 눈에 들어올 때 얼만큼 정보를 넘기는지인데, 이를 포함한 rendering 식을 MLP M으로 implicitly 예측
- f : geometry를 설계, M : appearance를 설계, 둘을 잇기 위해 가운데 layer 
- DVR과 IDR은 background가 noisy하면 surface 잘 못 찾아서 배경 없애는 mask 필요
- ray가 물체와 처음 만나는 점만 이용하는데 이를 UNISURF에서 해결

## UNISURF

- DVR과 IDR을 합친 느낌
- optimization할수록 delta_k 가 점점 줄어듬
- IDR 식 그대로 사용

## NeuS

- occupancy network 대신 SDF로 surface를 나타내자!  
SDF가 0이 되는 지점이 surface
  - IDR, NeuS, ...가 SDF 사용
  - eikonal loss 사용 가능해서 surface 잘 나타낼 수 있음  
  - PDF 정의 가능  
    - surface 근처에서 PDF 값이 크다
  - unbiased : 정확히 surface에서 weight가 locally maximal value를 가져야 한다
  - occlusion-aware : multiple surface인 상황에서 view-point과 더 가까운 surface가 color에 더 많은 기여를 해야 한다

## VolSDF

- logistic density distribution 대신 Laplace 로 SDF 만듬
- opacity O를 CDF라 생각하면 이를 미분한 tau(t)는 PDF이고 이를 rendering에 사용

## BakedSDF

- VolSDF + MipNeRF360
- VolSDF에서처럼 surface 찾기 -> marching cube로 mesh extraction -> spherical Gaussians로 appearance 표현