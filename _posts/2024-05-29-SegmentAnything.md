---
layout: post
title: SegmentAnything
date: 2024-05-29 14:00:00
description: Promptable Image Segmentation
tags: image segmentation
categories: cv-tasks
thumbnail: assets/img/2024-05-29-SegmentAnything/1m.PNG
giscus_comments: false
disqus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
images:
  compare: true
  slider: true
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

### SegmentAnything

#### Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick

> paper :  
[https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)  
출처 : Vision study mkd님  

---

## Abstract

- Task  
Promptable Image Segmentation  

- Model Architecture  
image encoder + prompt encoder + mask decoder  

- Generate Data (Data Engine)  
assisted-manual stage $$\rightarrow$$ semi-automatic stage $$\rightarrow$$ fully-automatic stage  
data 'SA-1B' : 1B masks with 11M images  

- Enable Zero-Shot Generalization  
Zero-Shot transfer to various tasks  

- Code Review


## Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/2m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- prompt : mask를 생성할 대상을 지정  
point, BB, mask(rough area), text(preliminary) 중 하나  

- valid masks : segmented mask를 하나가 아닌 3개 (whole, part, sub-part) 생성  
ambiguous prompt에 대응하기 위해, zero shot을 위해  
3개의 masks 중 GT와 가장 유사한(confidence score가 가장 높은) mask의 loss만 사용  


## Model

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/3m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Image Encoder : MAE (Masked AutoEncoder) 방식의 ViT  
MAE 요약 : 이미지를 grid로 나누고 patches 중 일부를 가린 뒤 원본을 복원하도록 학습하고, 학습이 끝난 후에는 encoder embedding만 사용  
ViT-H/16 : 14 $$\times$$ 14 windowed attention and 4 global attention blocks  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/4m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Prompt Encoder :  
Mask (dense prompt) : conv. 거친 후 image embedding에 pixel-wise sum (mask가 없는 pixel의 경우 'no mask' prompt 사용)  
Point (sparse prompt) : positional encoding + learned embedding(fg or bg)  
BB (sparse prompt) : positional encoding + learned embedding(top-left or bottom-right)  
text (sparse prompt) : by CLIP text encoder  

- Loss :  
1. Mask loss : related to mask prediction  
1-1. focal loss : $$L(p_{t}) = - (1-p_{t})^{r}log(p_{t})$$ where $$(1-p_{t})^{r}$$ gives more weight to few hard examples ($$p_{t} \sim 0$$)  
1-2. dice loss : 1 - dice score where dice score = $$\frac{2 \times Area(A \cap B)}{Area(A) + Area(B)}$$  

2. IoU loss : related to confidence score  
MSE loss  


## Data : Develop Data Engine by Curriculum Learning

- Assisted-manual stage :  
public segmentation dataset $$\rightarrow$$ SAM $$\rightarrow$$ pixel-wise manual augmentation $$\rightarrow$$ re-train  
After re-training, the number of masks per image increased from 20 to 44 in average  
Collect 4.3M masks from 0.12M images  

- Semi-automatic stage :  
dataset from previous stage (4.3M masks) $$\rightarrow$$ SAM $$\rightarrow$$ mask predict 실패한(제외된) object를 annotate $$\rightarrow$$ re-train  
After re-training, the number of masks per image increased from 44 to 72 in average  
Collect 5.9M masks from 0.18M images (totally 4.3M + 5.9M = 10.2M masks)  

- Fully-automatic stage :  
dataset from previous stage (10.2M masks) : image에 32 $$\times$$ 32 grid points 찍음 $$\rightarrow$$ SAM  
ambiguity-aware training (whole, part, sub-part 구분 가능)  
After filtering masks with high confidence score,  
Collect SA-1B dataset : 1.1B masks from 11M images (various HR masks)  
99.1% is fully-automatically generated  
follow RAI (Responsible AI) : no bias and blur human faces  


## Task

generalizable (zero-shot transfer to various tasks)  

- Zero-Shot Transfer Tasks :  
1. Zero-Shot Single Point Valid Mask Evaluation  
2. Zero-Shot Edge Detection
3. Zero-Shot Object Proposals
4. Zero-Shot Instance Segmentation
5. Zero-Shot Text-to-Mask (CLIP)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/4m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Zero-Shot Single Point Valid Mask Evaluation :  
point 찍었을 때 그에 해당하는 mask를 얼마나 잘 생성하는가  
use one most-confident mask  
compare with RITM model on 23 datasets  

- Zero-Shot Edge Detection :  
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/6m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
About filter : 블로그 맨 아랫 부분에 설명해놓음

- Zero-Shot Object Proposals :  
mask 예측 후 object의 identity(class)를 얼마나 잘 맞추는가  

- Zero-Shot Instance Segmentation :  
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/7m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Zero-Shot Text-to-Mask :  
image $$\rightarrow$$ CLIP $$\rightarrow$$ image embedding as input  
text $$\rightarrow$$ CLIP $$\rightarrow$$ text embedding as SAM prompt  
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/8m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

SAM's latent space에서 similar mask embedding vectors within threshold를 추출한 결과 실제로도 semantically similar
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/9m.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    A query is indicated by magenta box : top row shows matches at a low threshold and bottom row shows matches at a high threshold
</div>


## Code Review

다음에 해야지... 라고 미뤄둠..ㅎㅎ

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/10m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/11m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/12m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/13m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/14m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/15m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/16m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/17m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/18m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/19m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/20m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/21m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/22m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/23m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/24m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/25m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/26m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/27m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/28m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/29m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/30m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/31m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/32m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/33m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/34m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/35m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/36m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>  
</swiper-container>

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/37m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/38m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/39m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/40m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/41m.PNG" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>