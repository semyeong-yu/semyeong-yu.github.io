---
layout: distill
title: Multi-Modal Study
date: 2024-08-05 15:00:00
description: Paper Review
tags: multi-modal generative
categories: generative
giscus_comments: false
disqus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
# featured: true
toc:
  - name: StoryImager - A Unified and Efficient Framework for Coherent Story Visualization and Completion
  - name: Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models
  - name: Generating Realistic Images from In-the-wild Sounds
  - name: ViViT - A Video Vision Transformer
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

### StoryImager - A Unified and Efficient Framework for Coherent Story Visualization and Completion

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

### Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models

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
`현재에 가까운 과거 frame일수록 noise를 덜 주는 식으로 temporal order를 부여`하면 성능 좋음  

### Generating Realistic Images from In-the-wild Sounds

#### Taegyeong Lee, Jeonghun Kang, Hyeonyu Kim, Taehwan Kim

> paper :  
[Image from in-the-wild Sounds](https://arxiv.org/abs/2309.02405)

- novelty :  
이전까지는 wild sound와 image 간의 pair가 없어서 limited categories와 music의 sound로부터 image를 생성하는 연구만 진행되었음  
본 논문은 sound와 image 간의 large paired dataset이 없더라도  
wild sound로부터 image를 생성하는 task를 최초로 제시

- method :  
  - stage (a) :  
  `audio captioning`을 통해 sound를 text로 변환한 audio caption과  
  sound의 dynamic 특성을 반영하기 위한 `audio attention`과  
  제대로 image visualization하기 위한 `sentence attention`을  
  함께 사용하여 positional encoding을 거친 뒤 vector w를 `initialize`  
  (이 때, Audio-Captioning-Transformer model의 decoder에서 나오는 확률값을 audio attention이라고 정의함)
  - stage (b) :  
  audio caption으로부터 만든 vector z와  
  stage (a)의 vector w로부터  
  new latent vector z'를 만들고,  
  `stable-diffusion model`을 이용하여 이로부터 image를 생성한다  
  여기서 image와 vector z 간의 `CLIPscore similarity`를 이용해서 audio caption으로부터 만든 vector z를 optimize하고  
  image와 audio 간의 `AudioCLIP similarity`를 이용해서 `audio를 직접 optimize`한다  
  (image가 text에 맞게 생성되도록 image를 점점 변화시키면서 생성하는 Style-CLIP에서 영감을 받아 이를 diffusion model에 적용)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/ImageSound/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

local minimum에 빠지지 않기 위해 audio attention과 sentence attention을 이용한 stage (a)의 initialization이 매우 중요

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/ImageSound/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Q1 :  
image는 pixel 단위로 값이 있어서 feature map을 통해 vector로 만들 수 있고, text는 word 단위로 embedding을 통해 vector를 만들 수 있습니다. AudioCLIP을 통해 audio를 직접 optimize했다는데 audio는 무엇을 기준으로 vector로 만들어서 optimize 가능한 건가요? 

- A1 :  
audio는 melspectrogram을 만든 뒤 ViT에서 image 다루듯이 똑같이 patch로 쪼개서 vector로 만든다  
AudioCLIP similarity의 경우 audio encoding과 image encoding과 text encoding 간의 contrastive learning을 통해 구할 수 있다  


### ViViT - A Video Vision Transformer

#### Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, Cordelia Schmid

> paper :  
[ViViT](https://arxiv.org/abs/2103.15691)

- video of (T, H, W, C)를 sampling하여  
token sequence of (n_t, n_h, n_w, C) 을 만들고  
positional embedding을 더한 뒤 (N, d)로 reshape해서 transformer의 input으로 넣어줌

- uniform frame sampling :  
ViT에서처럼 각 2D frame을 독립적으로 embedding 후 모든 token을 concat

- Tubelet sampling :  
temporal info.를 반영하기 위해 tokenization 단계에서 spatial, temporal info.를 fuse

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/ViViT/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/ViViT/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Model 1 :  
CNN과 달리 transformer layer는 token 수에 비례하게 quadratic complexity를 가지므로 input frame에 linearly 필요

- Model 2 (factorized encoder) :  
spatial과 temporal을 두 개의 transformer encoder로 구성하여 
많은 연산량 필요

- Model 3 (factorized self-attention) :  
여전히 두 개의 encoder로 
특정 dim만 뽑아서 attention 연산

- Model 4 (factorized dot-product attention) :  
spatial head의 경우 spatial-dim.에 대해서만 attention 수행