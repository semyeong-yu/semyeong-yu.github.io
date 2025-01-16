---
layout: post
title: Object Detection
date: 2024-03-01 14:00:00
description: Detection
tags: vision detection
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

### Old Approach  

> Template matching with sliding window  

loss : MSE(SSD) / NCC / ZNCC (difference between 'image itself' and template)  

단점 : self-occlusions / change in appearance / change in position / change in scale or aspect ratio  

>  Viola-Jones detector (1-stage)  

>  HOG (Histogram of Oriented Gradients) (1-stage)  

> Proposals(RoIs) by selective search or edge box (2-stage)  

+) NMS에서 $$b_i$$와 비슷한 $$b_j$$들에 대해 confidence score를 비교하여 $$b_i$$ 제거 여부를 결정하는데, 만약 $$IoU_{threshold}$$를 넘겨야 비교 후 제거 (N) 가능하다면 high $$IoU_{threshold}$$ -> less FN, more FP  

### Detection Evaluation  

TP : positive(BB)라고 예측했는데 맞았다(true)  
FN : negative(no BB)라고 예측했는데 틀렸다(false)  

precision / recall / F1-score  

confindence score 순으로 P를 정렬한 뒤 AP(average precision)  
여러 object category에 대해 평균 낸 게 mAP  

### R-CNN (2-stage)  

> R-CNN : extract RoI -> crop&warp -> CNN -> SVM & BB reg  

장점 : CNN / transfer learning  

단점 :  
- slow (~2k proposals per image are warped and forwarded each through CNN)  
-> Fast R-CNN에서 SPP로 해결

- object proposal algorithm is flixed  
-> Faster R-CNN에서 RPN으로 해결         

- not end-to-end (CNN and SVM & BB reg are trained separately)  
-> Faster R-CNN에서 RPN으로 해결

> Fast R-CNN : CNN -> extract RoI -> SPP (RoI Pooling) -> fc -> classifier & BB reg  

SPP (= Spatial Pyramid Pooling) : fc layer 직전에 배치하면 any input size 가능  

R-CNN의 단점 1. 만 해결  

> Faster R-CNN :  CNN -> RPN (loss 1., 2.) -> RoI Pooling -> fc -> classifier & BB reg (loss 3., 4.)  

RPN (= Region Proposal Network) : output shape (H, W, 5n)  

- n anchors per location
- 1 confidence score 
- 4 normalized anchor coordinates

R-CNN의 단점 1., 2., 3. 모두 해결  

> FPN (= Feature Pyramid Network) :  

define RPN on each level of FPN  
scale variance 문제 해소  
high scale pyramid에서 small object까지 detect하므로 more TP, FP  
But, 단점 : model complexity

### 1-stage  

> YOLO (= You Only Look Once) : Faster R-CNN의 loss 3., 4.를 loss 1., 2.에 합치자!  

output shape (H, W, 5n) 대신 (S, S, (5+C)n)  

장점 : efficient, faster  
단점 : less accurate (coarse grid resolution)  
single scale (small object detect 불가능, scale variation에 취약)  

> SSD (= Single Shot Multibox Detector) : multi-scale을 사용하자!  

장점 : YOLO 단점 해결  
단점 : still less accurate than two-stage detectors due to class imbalance  
data augmentation 중요  

> class imbalance 문제 :  

two-stage detector의 경우 first stage에서 미리 negative anchor를 대부분 걸러낼 수 있지만  
one-stage detector는 그렇지 않아서 class imbalance 문제 발생  

대안 :  
- hard negative mining : FP 오류 줄이기 위해 어려웠던 sample들 추가  
- focal loss = $$-(1-p)^r * log(p)$$ : 많이 존재하는 easy example(p ~ 1)은 $$(1-p)^r$$로 영향 작게 만들고, 적게 존재하는 hard example(p ~ 0)에 가중을 둠  

> RetinaNet : 기존 1-stage 방법 + multi-scale(FPN) + focal loss  

> accuracy : YOLO < SSD < two-stage detector < RetinaNet  

> spatial transformer :  

grid generator로 sampling with bilinear interpolation  
= localisation & certain transformation  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-detection/img7m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-detection/img12m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-detection/img17m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-detection/img22m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-03-01-object-detection/img27m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>