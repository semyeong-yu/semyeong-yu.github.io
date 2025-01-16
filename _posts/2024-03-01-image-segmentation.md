---
layout: post
title: Image Segmentation
date: 2024-03-01 16:00:00
description: Segmentation
tags: vision segmentation
categories: cv-tasks
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  beginning: true
  sidebar: right
images:
  compare: true
  slider: true
---

뮌헨공대 (Technical University of Munich)에서 공부한 

[IN2375 Computer Vision - Detection, Segmentation and Tracking]
컴퓨터비전 노트 정리

### Semantic Segmentation

> K-means  

>  Spectral Clustering  

A : N x N similarity matrix  
$$D_{ii}$$ = $$sum_j$$($$a_{ij}$$)  
L = D - A  

L의 eigenvectors K개를 column으로 갖는 U에 대해 U의 rows를 data points로 보고 clustering  

>  CRF (Conditional Random Fields)  

useful post-processing tool (segmentation boundary 향상)  

- unary term : K[$$x_i$$ != $$y_i$$]  
- pairwise term : [$$x_i$$ != $$x_j$$]$$w_{ij}$$  

max-flow min-cut theorem으로 clustering  

단점 : unbalanced isolated clusters  

> Ncut (Normalized cut)  

minimize similarity b.w. different groups 뿐만 아니라 maximize similarity within each group까지 고려  

> FCN   

마지막에 fc layer (단점 : parameter 수 많고, fixed input size, no inductive bias) 대신  

1 x 1 conv로 semantic segmentation  

> high resolution

장점 : high segmentation accuracy  

단점 :  

- receptive field size 작음 -> dilated convolution으로 보완 (skip N-1 pixels in between, and then receptive field size = N(K-1)+1)  
- memory, FLOP 많이 소요 -> encoder - decoder 구조로 보완 (skip-connection 쓰면 better upsampling quality)  

> upsampling = transposed convolution = up-convolution  

!= deconvolution  

z = s - 1개의 0을 각 row, column 사이에 삽입 (unpooling)  
p' = k - p - 1개만큼 padding  
s' = 1으로 convolution  

문제 : checkboard artifacts -> 해결 : kernel size와 stride를 잘 선택 또는 interpolation(resize)한 뒤 conv  

### Instance Segmentation  

> proposal-based (Mask R-CNN)  

object detection -> segment and classify  

segmentation mask는 object detection보다 localization이 정확해야 하므로 RoI Pool 대신 RoI Align  

- RoI Pool : BB alignment -> bin alignment -> SPP  
- RoI Align : bilinear interpolation으로 feature value 계산  

+) Mask R-CNN 개선 by PointRend :  
bilinear upsampling 단점 : computation 많음 / boundary에서 fine detail 떨어짐  

-> 해결 : bilinear sampling (discrete pixel -> value) 대신  
trained network (continuous (x,y) -> value)로 refine boundaries  

> proposal-free (SOLOv2)  

semantic segmentation -> group pixels  

instance 개수 몇 개일지 모르므로 predict the kernels G : S x S x D  

### Panoptic Segmentation  

> proposal-based (Panoptic FPN)  

1. FPN  
2. decoder CNN (semantic segmentation) 및 Mask R-CNN (instance segmentation)  
3. 합침  

합칠 때 thing 우선이므로  

사실은 thing이라서 "other"로 labelling된 stuff 또는 small region stuff는 제거  

instance(class / BB reg / mask) / semantic 으로 총 4가지의 loss term  

> proposal-free (Panoptic FCN)  

SOLOv2에서처럼 predict the kernels  

1. FPN  
2. position head ($$N_{thing} x H_i x W_i$$ / $$N_{stuff} x H_i x W_i$$) 및 kernel head ($$C_{in} x Hi x Wi$$)로 kernel fusion $$(N_{thing}+N_{stuff}) x C_in x 1 x 1$$  

> Panoptic Segmentation Evaluation  

PQ = SQ x RQ = MOTP x F1-score  
under unique matching theorem  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-image-segmentation/img52m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-image-segmentation/img57m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-image-segmentation/img62m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-image-segmentation/img67m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>