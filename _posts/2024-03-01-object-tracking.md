---
layout: post
title: Object Tracking
date: 2024-03-01 15:00:00
description: Tracking
tags: vision tracking
categories: cv-tasks
giscus_comments: true
related_posts: true
toc:
  beginning: true
  sidebar: right
---

뮌헨공대 (Technical University of Munich)에서 공부한  
[IN2375 Computer Vision - Detection, Segmentation and Tracking]
컴퓨터비전 노트 정리  

### General Bayesian Framework

> setting

$$x_k$$ : internal state, hidden random var. e.g. BB  

$$z_k$$ : measurement, observable random var. e.g. image data  

$$X_k$$ = [$$x_1$$, ..., $$x_k$$]  

$$Z_k$$ = [$$z_1$$, ..., $$z_k$$]  

>  assumptions

likelihood p($$z_k \vert x_k$$)는 $$Z_{k-1}$$ 에 무관  

temporal prior p($$x_k \vert x_{k-1}$$)는 $$Z_{k-1}$$ 또는 $$X_{k-2}$$ 에 무관  

>  Bayesian framework

아래 사진에 있는 식과 같이 posteior p($x_k$ $\vert$ $Z_k$)를 recursively 구할 수 있음  

posterior mean = E($$x_k \vert Z_k$$)  

MAP(maximum a posterior) = $$argmax_{x_k} (p(x_k \vert Z_k))$$  

+) deep learning : learn MAP directly  

- online tracking : computational overhead / drifting에 취약  
- offline tracking : real-time (X) / 새로운 frame에 적응 (X)  

### Single Object Tracking (online)

> GOTURN

assumption : MAP at frame k approximates to $$x_k$$  
(즉, use position of BB in frame k-1 to crop in frame k)

장점 : simple -> real-time / end-to-end (can use large data)  
단점 : simple temporal prior -> one object possible / fast motion or occlusion (X) / template에 의존 

> MDNet

아이디어 : 매 training sequence(task)마다 domain-specific fc layers 따로 만들면 inefficient하므로 online adaptable fc layer 사용하여 appearance change에 적응  

architecture : 매 frame마다 아래의 과정 반복  
candidates -> R-CNN으로 optimal state 찾기 -> bias 추가로 collect pos/neg samples -> fine-tuning으로 update fc layer  

장점 : fine-tuning / online adaptable appearance model이므로 change in appearance에 적응  

단점 : slow / strong assumption on temporal prior  

### Multiple Object Tracking (online)  

> Match

predict next position using motion model (Kalman filter / LSTM, GRU)  
match (Hungarian matching O(N^3))

> Refine (Tracktor)

매 frame마다 아래의 과정 반복  
copy boxes to frame k -> regression (refine boxes) & classification (kill if low conf) & detection (find new BB)  

장점 : object detector 재사용 가능 / still image로 훈련된 object detector여도 괜찮 / regression is agnostic to ID or class  

단점 :  
- no motion model, so fast motion or occlusion에 취약 -> long term memory of tracks로 보완  
-  identity 없으므로 crowded space에 취약

> Metric learning for Re-ID 

small motion assumption이 필요한 motion model (IoU) 말고 more robust "appearance" model을 만들어보자!  

contrastive learning : positive & negative pair 만들어서 siamese network를 통해 hinge 또는 triplet loss  

### Multiple Object Tracking (offline) : Graph-based  

> setting  

- goal : find maximum flow with minimum cost

- Bayesian framework로부터 cost 유도하기 : 아래 사진 참고  
likelihood -> detection cost  
prior -> entrance / transition / exit cost  

- Markov formulation은 occlusion 설명 불가능 / not end-to-end

> Message Passing Network

- initialization :  
node : from BB  
edge : from BB coordinates 차이 및 reid distance 차이  

- neural message passing :  
node-to-edge :  
($$node_i$$ at k-1), ($$node_j$$ at k-1), ($$edge_{ij}$$ at k-1), ($$edge_{ij}$$ at 0) -> $$edge_{ij}$$ at k  

- edge-to-node :  
($$edge_{ij}$$ at k for every neighbor j), ($$node_i$$ at k-1) -> $$node_i$$ at k  
이 때, node-permutation-invariant 이므로 set or sum 사용  

- edge classification : 단순히 thresholding 뿐만 아니라 post-processing 필요  

- loss :  
아래 사진에서 w > 1 이면 gt = 1 (active adge)일 때 틀리면 안 된다는 의미이므로 penalize FN  

장점 : handle occlusions / end-to-end (cost is learned) / graph structure 자유 / SOTA for offline  

> Multiple Object Tracking Evaluation

MOTA / F1-score / MOTP  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-tracking/img27.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-tracking/img32.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-tracking/img37.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-tracking/img42.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-tracking/img47.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>
