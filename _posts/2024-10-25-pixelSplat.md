---
layout: distill
title: pixelSplat
date: 2024-10-25 12:00:00
description: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction (CVPR 2024)
tags: 3DGS image pair scalable
categories: 3d-view-synthesis
thumbnail: assets/img/2024-10-25-pixelSplat/1.PNG
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  - name: Introduction
  - name: Background
  - name: Methods
  - name: Experiments
  - name: Conclusion

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

## pixelSplat - 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction (CVPR 2024)

#### David Charatan, Sizhe Li, Andrea Tagliasacchi, Vincent Sitzmann

> paper :  
[https://arxiv.org/abs/2312.12337](https://arxiv.org/abs/2312.12337)  
project website :  
[https://davidcharatan.com/pixelsplat/](https://davidcharatan.com/pixelsplat/)  
code :  
[https://github.com/dcharatan/pixelsplat](https://github.com/dcharatan/pixelsplat)  
reference :  
NeRF and 3DGS Study

### Introduction

- Problem :  
  - `scale ambiguity` :  
  camera pose has arbitrary scale factor
  - `local minima` :  
  primitive param.을 random initialization으로부터 직접 optimize하면 local minima 문제 발생

- Solution :  
  - `feed-forward model` 이 `a pair of images`로부터  
  `3DGS primitives`로 parameterized되는 3D radiance field recon.을 학습  
  
### Background

- local minima :  
  - random location에 initialization된 Gaussian primitives가  
  최종 목적지까지 a few std보다 더 많이 움직여야 될 때  
  또는  
  최종 목적지까지 loss가 monotonically decrease하지 않을 때  
  local minima 발생
  - 이를 해결하기 위해 3DGS에서는  
  non-differentiable pruning and splitting 기법인  
  Adaptive Density Controld을 사용하지만  
  본 논문에서는  
  differentiable parameterization of Gaussian primitives 소개

### Methods

### Experiments

### Conclusion
