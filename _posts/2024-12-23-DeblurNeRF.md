---
layout: distill
title: Deblurring NeRF Summary
date: 2024-12-23 12:00:00
description: brief summary
tags: 3DGS deblur
categories: 3d-view-synthesis
thumbnail: assets/img/2024-12-23-DeblurNeRF/1.PNG
giscus_comments: false
disqus_comments: true
related_posts: true
bibliography: 2024-12-23-DeblurNeRF.bib
toc:
  - name: DoF-NeRF
  - name: Deblur-NeRF
  - name: DP-NeRF
  - name: PDRF
  - name: Hybrid
  - name: Sharp-NeRF
  - name: BAD-NeRF

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

## Deblurring NeRF Summary

### DoF-NeRF

> reference :  
[DoF-NeRF](https://jseobyun.tistory.com/301)


<d-cite key="DofNeRF">[1]</d-cite>

단점 :  
train하기 위해 all-in-focus image와 blurry image 모두 필요  
(all-in-focus image : 화면 전체가 초점이 맞춰져 있는 image)

### Deblur-NeRF

<d-cite key="DeblurNeRF">[2]</d-cite>

장점 :  
train할 때 all-in-focus image 필요 없음  
핵심 :  
additional small MLP 사용해서  
per-pixel blur kernel 예측

### DP-NeRF

<d-cite key="DpNeRF">[3]</d-cite> 

### PDRF

<d-cite key="PDRF">[4]</d-cite>

### Hybrid

Hybrid <d-cite key="Hybrid">[5]</d-cite>

camera motion blur와 defocus blur 중 하나만 다룸

### Sharp-NeRF

<d-cite key="SharpNeRF">[6]</d-cite>

camera motion blur와 defocus blur 중 하나만 다룸

### BAD-NeRF

<d-cite key="BADNeRF">[7]</d-cite>

camera motion blur와 defocus blur 중 하나만 다룸

### Future Work

- NeRF (implicit, stochastic method) 자체가 rendering이 오래 걸려서  
3DGS Deblurring도 많이 연구되고 있음!