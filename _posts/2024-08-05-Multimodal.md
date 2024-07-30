---
layout: distill
title: Multi-Modal Study
date: 2024-08-05 15:00:00
description: Paper Review
tags: multi-modal generative
categories: generative
thumbnail: assets/img/2024-08-05-Multimodal/1.png
giscus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
# featured: true
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

## Multi-Modal Study

### StoryImager: A Unified and Efficient Framework for Coherent Story Visualization and Completion

#### Ming Tao, Bing-Kun Bao, Hao Tang, Yaowei Wang, Changsheng Xu

> paper :  
[StoryImager](https://arxiv.org/abs/2404.05979)  

- `Story Visualization` Task :  
  - input : prince image / cat image / prompts
  - output : story images
  - video generation과는 다른 게, story visualization은 웹툰 같다고 생각하면 됨  
  story visualization은 image 간의 consistency를 유지하긴 하지만, video generation처럼 frame끼리 연속성을 보장할 필요는 없음

- StoryImager:  
  - task : coherent story visualization and completion
  - 기존 모델은 visualization과 continuation을 위한 model을 별도로 필요했는데,  
  본 논문은 single model (`통합적인 framework`) 제시  
  by `global consistency` 반영!

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/StoryImager/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

- Architecture :  
maintain `global consistency`  
by context-feature-extractor  
and FS-CAM (frame-story cross-attention module)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/StoryImager/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

- `random masking` :  
make a story board from story images  
$$\rightarrow$$ VAE  
$$\rightarrow$$ random masking to VAE latent space 

- `context-feature-extractor` :  
  - word-embeddings $$\rightarrow$$ transformer  
  $$\rightarrow$$ prior embeddings  
  $$\rightarrow$$ MLP  
  $$\rightarrow$$ frame-aware latent prior  
  $$\rightarrow$$ `story board images가 story prompts를 반영하도록` 하기 위해 masked VAE latent space와 concat해서 이를 FS-CAM에서 story board로 사용  
  - word-embeddings $$\rightarrow$$ transformer  
  $$\rightarrow$$ context embeddings  
  $$\rightarrow$$ transformer  
  $$\rightarrow$$ global context  
  $$\rightarrow$$ `global 정보` 주기 위해 FS-CAM에서 text prompts로 사용

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/StoryImager/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

- FS-CAM (`frame-story cross-attention module`) :  
개별 story board - 개별 text prompts 를 locally cross-attention한 것과,  
전체 story board - 전체 text prompts 를 globally cross-attention한 것을  
concat

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/StoryImager/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

### Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models

#### Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, Weidi Xie

> paper :  
[Intelligent Grimm](https://arxiv.org/abs/2306.00973)  

- Intelligent Grimm :  
  - NIPS 2023에서 novelty 부족으로 reject 당했다가 보완해서 CVPR 2024에 accept  
  - task : open-ended visual storytelling  
  - StoryGen : unseen characters에 대해서도 추가적인 character-specific-optimization 없이 story visualization 가능  
  - StorySalon : online-video, open-source e-book 등 소싱해서 만든 dataset

- `context encoding` :  
  - SDM : pre-trained stable diffusion model  
  CLIP : pre-trained CLIP encoder  
  VAE : pre-trained VAE  
  - visual condition feature = [SDM(image1, CLIP(text1)), SDM(image2, CLIP(text2)), ..., SDM(imagek-1, CLIP(textk-1))]  
  k-th frame image 만들기 위해 cross-attention 하는 데 사용  

- visual-language contextual fusion :  
`cross-attention` 사용  
아래 Fig. (b)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/IntelligentGrimm/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

- `conditional generation` :  
  - prev. frame t-1 $$\rightarrow$$ add noise  
  $$\rightarrow$$ denoising one step  
  $$\rightarrow$$ diffusion U-Net  
  - prev. text $$\rightarrow$$ text encoder  
  $$\rightarrow$$ diffusion U-Net  
  - diffusion U-Net (self-attention, text-attention)  
  $$\rightarrow$$ `denoising one step에 해당되는 feature`를 추출  
  - extracted image-diffusion-denoising-feature  
  & random noise  
  & current text $$\rightarrow$$ text encoder  
  $$\rightarrow$$ StoryGen model (self-attention, image-attention, text-attention)  
  $$\rightarrow$$ current frame t  

- `multi-frame conditioning` :  
story의 경우 frame t에 영향을 주는 image들이 frame t-1, frame t-2, ... 일 수 있음  
이 때, 과거 frames에 모두 같은 random noise를 줄 경우 성능 좋지 않아서  
`현재에 가까운 과거 frame일수록 noise를 덜 주는 식으로 temporal order를 형성`하면 성능 좋음  

### ???

#### ???

> paper :  
[???](???)

- ???