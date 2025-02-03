---
layout: distill
title: NoPoSplat
date: 2025-02-03 10:00:00
description: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (ICLR 2025)
tags: dynamic GS SfMfree
categories: 3d-view-synthesis
thumbnail: assets/img/2025-02-03-NoPoSplat/1.PNG
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  - name: Contribution
  - name: Related Works
  - name: Method
    subsections:
      - name: Architecture
      - name: Gaussian Space
      - name: Camera Intrinsic Embedding
      - name: Training and Inference
  - name: Experiment
    subsections:
      - name: Implementation
      - name: Result
      - name: Ablation Study
  - name: Conclusion
  - name: Question
bibliography: 2025-02-03-NoPoSplat.bib
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

## No Pose, No Problem - Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images

#### Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, Songyou Peng

> paper :  
[https://arxiv.org/abs/2410.24207](https://arxiv.org/abs/2410.24207)  
project website :  
[https://noposplat.github.io/](https://noposplat.github.io/)  
code :  
[https://github.com/cvg/NoPoSplat](https://github.com/cvg/NoPoSplat)

## Contribution

`pose-free generalizable 3D recon.!`

- inference :  
  - `unposed sparse` images로부터 3DGS를 통해 3D scene recon.
  - recon.된 3DGS를 이용하여 novel-view-synthesis 및 pose-estimation task 수행 가능  
  특히 limited input image overlap (sparse) 상황에서는 pose-required methods보다 더 좋은 성능

- Gaussian Space :  
  - `하나의 input view의 local camera coordinate`을 `canonical space`로 잡고 해당 space에서 3DGS 표현
  - local coordinate에서 `global coordinate으로 3DGS를 변환할 필요가 없음`  
  $$\rightarrow$$  
  per-frame Gaussians 및 pose estimation에서 유래되는 error 없음
  - pose 없이도 3D recon. 가능

- training :  
  - `scale ambiguity` 문제 해결을 위해  
  `intrinsic embedding method` 사용
  - GT depth나 explicit matching loss 사용 X
  - two-stage coarse-to-fine pipeline 제시

## Related Works

- MonST3R :  
  - 차이점 1)  
  MonST3R는 transformer의 output이 3D pointmap(pcd)인데,  
  NoPoSplat은 mean 뿐만 아니라 scale, rotation, opacity, color를 가진 3D Gaussian (rasterization) 사용
  - 차이점 2)  
  MonST3R는 pairwise pointmaps를 global pointmap으로 변환하는 $$P$$ 를 학습하는데,  
  NoPoSplat은 global coordinate으로 3DGS를 변환할 필요 없음 `???`

- pixelSplat :  
  - 차이점 1)  
  inference할 때 pixelSplat은 2D-to-3D로 unproject하여 3D Gaussian 만들고자 ray direction $$d = R K^{-1} [u, 1]^{T}$$ 에서 camera pose를 input으로 사용하는데,  
  inference할 때 NoPoSplat은 `???` TBD

## Method

### Architecture

### Gaussian Space

### Camera Intrinsic Embedding

### Training and Inference

## Experiment

### Implementation

### Result

### Ablation Study

## Conclusion

## Question