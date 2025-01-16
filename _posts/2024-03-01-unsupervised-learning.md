---
layout: post
title: Unsupervised Learning
date: 2024-03-01 20:00:00
description: Unsupervised Learning
tags: vision unsupervised
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

### Unsupervised Learning  

특징 : fast / costly (much computation)  

### Self-supervised learning  

>  Evaluation  

fine-tuning / linear probing / k-NN classification  

> Pretext task  

사용자가 직접 정의한 새로운 different task (goal task와 관련 있어야 함)  

> Contrastive Learning  

아이디어 :  

positive pair (image vs augmented image)와 negative pair (image vs other image) 만들어서 triplet loss 사용 또는 maximize contrastive score (아래 사진 참고)  

- SimCLR (Simple framework for Contrastive Learning) :  

two augmented images에 대해 maximize similarity b.w. feature maps  

장점 : simple  

단점 : negative samples 수가 많을수록 gradient bias가 줄어드므로  
large batchsize가 필요한데 그러면 large computation 필요  

- MoCo (Momentum Contrast) :  

1. memory queue를 통해 GPU memory footprint 작아서 large batchsize 가능  
2. dynamic update of momentum encoder로 성능 향상  

> Non-Contrastive Learning  

other images (negative pair) 쓰지 않음  

- DINO (self-distillation with no labels) :  
ViT 등에서 robust latent embedding 만드는 데 사용  

entering : teacher sharpening할 때 collapse되는 것을 방지하기 위해 moving average of center를 빼줌  

sharpening : teacher가 student보다 lower temperature로 sharpened  

- DINOv2 :  

DINO보다 2배 faster, 3배 less memory  by  noisy student, adaptive resolution, data curation, ...  

- MAE (Masked Autoencoders) :  

high masking ratio 필요  

masked patches를 reconstruct한 뒤 이에 대해서만 loss 계산  

>  multi-view assumption  

view(crop) provides enough info. for downstream task  

>  PAC (pixel-adaptive conv.) layer  

conv.를 하기 전에 spatially varying kernel을 곱함  

### Downstream Applications  

> semantic segmentation 에서 KNN correspondence  

two image에 대해 "segmentation correspondence"가 "feature(DINO) correspondence"를 모방하도록 학습  

> motion-based VOS  

network G : image & optical flow -> object mask  

network I : masked optical flow & image -> reconstruct optical flow  

> Contrastive Random Walk (object tracking)  

아래 사진 참고.  

cycle consistency loss (cross-entropy loss b.w. label_t=0 and label_t=T)  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-unsupervised-learning/img97m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-unsupervised-learning/img102m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-unsupervised-learning/img107m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-unsupervised-learning/img112m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>
