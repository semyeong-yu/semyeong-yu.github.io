---
layout: post
title: Semi-Supervised Learning
date: 2024-03-01 19:00:00
description: Semi-Supervised Learning
tags: vision semisupervised
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

### Semi-supervised Learning

>  아이디어  

use both labelled and unlabelled data  

> assumption

- smoothness : if two inputs are close, their labels are same  
- low density : decision boundary should pass through region with low density  
- manifold : data come from multiple low-dim. manifolds if data points share same manifold, their labels are same  

> unsupervised pre-processing  

feature extraction  

> wrapper method 중 self-training  

OnAVOS는 slow이므로 first frame/new frame 대신  
offline으로 labelled/unlabelled data 사용  

initial prediction이 중요하므로 미리 train strong baseline on labelled set  

>  energy minimization (low-density assumption 적용)  

minimize $$-p(x_i)logp(x_i)$$ = entropy of class distribution of each pixel $$x_i$$  

>  VAN (virtual adversarial network) (smoothness assumption 적용)  

labelled set : true posterior(gt) 와 adversarial 추가한 image의 prediction 비교  

unlabelled set : 기존 image의 prediction 과 adversarial 추가한 image의 prediction 비교  

> Domain Alignment  

GAN의 원리 사용하여 unlabelled real data와 labelled synthetic data의 distribution을 비슷하게  

> Consistency Regularization  

image에 transformation을 가하더라도 robust하게 consistent prediction을 하도록 consistency loss 추가  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-semisupervised-learning/img112m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-semisupervised-learning/img117m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-semisupervised-learning/img122m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>
