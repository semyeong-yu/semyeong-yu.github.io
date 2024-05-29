---
layout: distill
title: SegmentAnything
date: 2024-05-29 14:00:00
description: Image Segmentation
tags: image segmentation
categories: cv-tasks
thumbnail: assets/img/2024-05-29-SegmentAnything/2.png
giscus_comments: true
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

> paper :  
[https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)  

---

## Abstract

- Enable zero-shot generalization  
NLP에서 next token generation 했듯이 promptable segmentation  

- Corresponding model architecture  
image encoder + prompt encoder + mask decoder  

- Generate data  
data engine : assisted manual generation $$\rightarrow$$ semi-automatic generation $$\rightarrow$$ fully-automatic generation  
data 'SA-1B' : 1B masks with 11M images  


## Task

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- prompt : mask를 생성할 대상을 지정  
point, BB, mask(rough area), text(preliminary) 중 하나  

- valid masks : mask를 하나가 아닌 3개 생성  
ambiguous prompt에 대응하기 위해, zero shot을 위해  


## Model

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-29-SegmentAnything/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Image Encoder : MAE (Masked AutoEncoder) 방식의 ViT  
MAE : 이미지를 grid로 나누고 patches 중 일부를 가린 뒤 원본을 복원하도록 학습하고, 학습이 끝난 후에는 encoder embedding만 사용  
ViT-H/16 : 14 $$\times$$ 14 windowed attention and 4 global attention blocks  

- Prompt Encoder :  
Mask (dense prompt) : mask가 없는 경우 'no mask' prompt 사용  
Point (sparse prompt) : sum of positional encodings  
BB (sparse prompt) : ddd  
text (sparse prompt) : ddd  

- Mask Decoder :  
ddd 


## Data

- ddd
