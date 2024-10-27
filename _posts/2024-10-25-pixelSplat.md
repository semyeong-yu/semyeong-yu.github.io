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
  - name: Scale Ambiguity
  - name: Gaussian Parameter Prediction
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

- Contribution :  
  - `two-view image encoder` :  
  scale ambiguity 문제 극복
  - `pixel-aligned Gaussian param. prediction module` :  
  local minima 문제 극복
  
- Solution :  
  - `feed-forward model` 이, `a pair of images`로부터,  
  `3DGS primitives`로 parameterized되는 3D radiance field recon.을 학습  

- overview :  
  - input :  
  a pair of images  
  associated camera parameters  
  - task :  
  3DGS representation of 3D scene

### Background

- local minima :  
  - 언제 발생? :  
  random location에 initialize된 Gaussian primitives가  
  최종 목적지까지 a few std보다 더 `많이 움직`여야 될 때  
  gradient가 vanish 되어버려서  
  또는  
  최종 목적지까지 loss가 monotonically decrease하지 않을 때  
  local minima 발생
  - 해결법? :  
  3DGS에서는  
  non-differentiable pruning and splitting 기법인  
  Adaptive Density Control을 사용하지만  
  본 논문에서는  
  `differentiable` parameterization of Gaussian primitives 소개

### Scale Ambiguity

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-10-25-pixelSplat/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Scale Ambiguity 문제 :  
  - `SfM 단계`에서 camera pose를 계산할 때  
  real-world-scale pose $$T_{j}^{m}$$ 을  
  metric pose $$s_{i} T_{j}^{m}$$ 으로 scale하여 사용  
    - $$s_{i}$$ :  
    arbitrary scale factor  
    - metric pose $$s_{i} T_{j}^{m}$$ :  
    real-world-scale pose의 translation component를 $$s_{i}$$ 만큼 scale
  - single image의 camera pose $$s_{i} T_{j}^{m}$$ 로부터  
  arbitrary scale factor $$s_{i}$$ 를 복원하는 건 불가능

- Scale Ambiguity 해결 :  
  - two-view encoder에서 `a pair of images` 로부터  
  arbitrary scale factor $$s_{i}$$ 복원

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-10-25-pixelSplat/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Two-view encoder Overview :  
  - Step 1) Per-Image Encoder  
  - Step 2) Epipolar Sampling  
  - Step 3) Epipolar Attention  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-10-25-pixelSplat/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Step 1) Per-Image Encoder
</div>

- Step 1) Per-Image Encoder :  
each view (two images)를 각각 feature $$F$$, $$\tilde F$$ 로 encode

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-10-25-pixelSplat/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Step 2) Epipolar Sampling
</div>

- Step 2) Epipolar Sampling :  
TBD 10p

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-10-25-pixelSplat/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Step 3) Epipolar Attention
</div>

- Step 3) Epipolar Attention :  
TBD 11p
per-pixel correspondence 찾고,  
해당 pixel에 대응되는 depth 기억

### Gaussian Parameter Prediction

### Experiments

### Conclusion
