---
layout: distill
title: Neuralangelo
date: 2024-09-30 12:00:00
description: High-Fidelity Neural Surface Reconstruction (CVPR 2023)
tags: 3d surface hash grid numerical gradient
categories: 3d-view-synthesis
thumbnail: assets/img/2024-09-30-Neuralangelo/1.png
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  - name: Abstract
  - name: Related Works
  - name: Numerical Gradient
  - name: Coarse-to-Fine
  - name: Question
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

## Neuralangelo - High-Fidelity Neural Surface Reconstruction (CVPR 2023)

#### Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, Chen-Hsuan Lin

> paper :  
[https://arxiv.org/pdf/2306.03092](https://arxiv.org/pdf/2306.03092)  
project website :  
[https://research.nvidia.com/labs/dir/neuralangelo/](https://research.nvidia.com/labs/dir/neuralangelo/)  
code :  
[https://research.nvidia.com/labs/dir/neuralangelo/](https://research.nvidia.com/labs/dir/neuralangelo/)  
reference :  
NeRF and 3DGS Study

### Abstract

### Related Works

- Instant NGP: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding  
(Website)[https://nvlabs.github.io/instant-ngp/] (SIGGRAPH 2022)

### Numerical Gradient

- numerical gradient : $$\text{lim}_{\epsilon \rightarrow 0} \frac{\partial }{}$$  
analytical gradient : $$\frac{\partial y}{\partial x}$$

- multi-resolution hash grid :  
  - 문제 :  
  SDF-based volume rendering에 `multi-resolution hash grid` 을 직접적으로 적용하면  
  large smooth regions의 surface에 noise 및 hole이 생김
  - 이유 :  
    - locality!
    - first-order derivative of SDF는 Eikonal constraints on the surface normals 계산을 위해 사용되고  
    second-order derivative of SDF는 surface curvatures 계산을 위해 사용됨  
    그리고 이러한 higher-order derivative의 de-facto approach는 `analytical gradient`를 사용하므로

### Coarse-to-Fine

### Question