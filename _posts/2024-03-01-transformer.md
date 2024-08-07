---
layout: post
title: Transformer
date: 2024-03-01 18:00:00
description: Transformer
tags: vision transformer
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

### Ideas in Transformer  

> positional encoding  

sequential model에서는 위치 정보를 temporal 차원으로 펼쳐서 나타내서 vanishing gradient 등의 문제가 생기는데, transformer는 "positional encoding"이라는 하나의 module에서 위치 정보를 부여할 수 있다!  

> attention  

input data를 "한 번에 넣어서" 어디에 더 가중치를 부여할 지 계산!  

> multi-head attention  

attention 구조는 auto-regressive 하지 않으므로 dimension을 쪼개서 "efficient parallel computation"을 가능하도록 하는 구조!  

### Attention  

> transformer  

learn non-local (global) info. in the same layer in parallel  

>  self-attention  

$$Q = XW_q$$ : 질문  

$$K = XW_k$$ : 답변  

$$V = XW_v$$ : 표현  

$$Y = softmax(QK^T/sqrt(n))V W_o$$  

$$softmax(QK^T/sqrt(n))$$ : query와 key가 얼마나 유사한지에 대한 attention weight 그 자체 : 어디를 봐야 돼?  
$$V$$ : weighted-sum을 할 단어 : 어떤 단어가 와야 돼?  

softmax 안의 값이 너무 커지면 vanishing gradient 문제 발생하므로 sqrt(n)으로 나눔  

memory complexity : $$O(T^2)$$ / runtime complexity : $$O(nT^2)$$  

> multi-head attention  

Q, K, V의 dimension을 Z개로 쪼개서 따로 attention한 뒤 concat으로 합치고 $$W_o$$ 곱함  

장점 :  
- 쪼개서 attention 여러 번 하므로 more complex interaction 포착 가능  
- runtime complexity는 그대로지만 parallel하게 수행하므로 실질적으론 less computation / faster  

### Transformer

> ViT (Vision Transformer)  

for classification  

extra [class] embedding 추가한 뒤 positional encoding of embedded patches  

단점 : ViT는 CNN의 inductive bias (locality / 2D neighborhood structure / translation invariance) 없으므로 must be pre-trained on large JFT dataset  

> DETR (Deformable Transformer)  

for detection  

1. transformer encoder with CNN features  

2. transformer decoder with object queries (= learnable positional encoding)  

3. predict class & BB in parallel  

+) positional encoding은 Q, K에 적용  

+) decoder에서 object queries는 Q 역할, encoder output은 K, V 역할  

loss :  

- BB : no object가 아닌 BB에 대해 (1 - gIoU) + (L1 loss)  
+) gIoU : IoU=0이어서 vanishing gradient 문제 발생하는 것 방지  

- class : cross-entropy loss  

장점 : accurate / simple / NMS 필요 X  
단점 : computation, memory 소모 / slow  

> Swin Transformer

아이디어 : computation 이득을 위해 # of tokens (resolution)을 줄여보자  

1. 처음부터 여러 resolution의 hierarchy를 만들자  
2. low resolution의 경우 alternate the layout of local attention windows 로 aggregate global context  
3. stage 지날 때마다 patch merging 으로 reduce # of tokens  

장점 :  
efficient, accurate (SOTA)  
CNN의 inductive bias + Transformer 이므로 ViT와 달리 미리 pre-train할 필요 X  

> MaskFormer

Panoptic FCN처럼 compute kernel(mask), but  
Panoptic FCN과 달리  
learnable queries with Transformer로 kernel(mask) 계산  

stuff, thing 이라는 개념 대신 N개의 queries  

단점 : small object일 때 잘 작동 X  

> Mask2Former  

pixel decoder에 있는 multiple scale의 feature map을 Transformer decoder에 사용하여 SOTA  

> masked attention  

N개의 learnable query를 이용해서 predict mask of shape (N, H, W), and apply masked-attention (fg area에 대해서만 attention 하기 위해) : masked localized feature 만으로 attention 해도 충분하고 오히려 더 efficient  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-transformer/img82.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-transformer/img87.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-transformer/img92.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>