---
layout: post
title: Video Segmentation
date: 2024-03-01 17:00:00
description: vos(video object segmentation)
tags: vision video segmentation
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

### Motion-based VOS  

> Motion-based VOS  

장점 : fast, accurate to obtain optical flow  
단점 : optical flow만으로는 occlusion 및 illumination change에 취약  

>  assumption  

- brightness constancy : brightness in the small region remains the same  
- spatial coherence : neighboring points in 2D are in same surface, so have similar 3D moton  
- temporal persistence : motion changes gradually over time  

+) assumption 1., 2. 이용해서 classic Lucas-Kanade method로 optical flow motion (u, v) 구할 수 있음  

>  FlowNet (optical flow)  

- architecture : two images를 siamese network에 넣음  
- correlation layer : spatial similarity between two feature maps  
- refinement : deconv. 만 하는 게 아니라 upsampled prediction과 skip-connection을 더함 for high-quality upsampling  

> SegFlow (joint of optical flow and segment)  

optical flow output과 segmentation output에 대해 둘 중 하나는 freeze하여 번갈아가며 optimize  

### Appearance-only VOS

> Appearance-based VOS  

- 장점 : appearance라 static image로 훈련 가능 / optical flow와 달리 occlusion에 대응 가능 / simple  
- 단점 : temporal consistency (X) / slow (model을 each frame에 independently 적용 at test)  

> Appearance-only VOS

아이디어 : model을 each frame에 independently 적용 at test  

> OSVOS (One-shot VOS)

test sequence의 "first frame"에 대해 fine-tune 하여 learn which object to segment  

단점 : drifting (appearance change에 적응 불가능)  

> OnAVOS (Online Adaptation VOS)

test sequence의 "every frame"에 대해 fine-tune 하여 online adaptation to appearance change  

every frame으로부터 pseudo-label을 추출하여 annotated sample에 추가  

장점 : appearance change에 적응 가능, so more accurate  
단점 : slow / pseudo-label may be inaccurate  

> MaskTrack  

MOT의 Tracktor와 비슷한 원리  

perturb mask by affine transformation or non-rigid deformation -> recover original mask  

### Metric-based VOS (Appearance-based)

> Pixel-wise Retrieval  

training 필요 없이 unsupervised method로 fast  

triplet loss로 fg pixel끼리는 가까이, bg pixel은 멀리 있도록 learn pixel-level embedding  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-video-segmentation/img72.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-video-segmentation/img77.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-video-segmentation/img82.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>
