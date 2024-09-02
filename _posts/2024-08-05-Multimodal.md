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
  - name: LLaMA-VID - An Image is Worth 2 Tokens in Large Language Models
  - name: PEEKABOO - Interactive Video Generation via Masked-Diffusion
  - name: Style Aligned Image Generation via Shared Attention
  - name: ControlNet - Adding Conditional Control to Text-to-Image Diffusion Models
  - name: InstructPix2Pix - Learning to Follow Image Editing Instructions
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

## StoryImager - A Unified and Efficient Framework for Coherent Story Visualization and Completion

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

## Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models

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

## Generating Realistic Images from In-the-wild Sounds

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


## ViViT - A Video Vision Transformer

#### Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, Cordelia Schmid

> paper :  
[ViViT](https://arxiv.org/abs/2103.15691)

**ViViT는 아직 정리 완료 못했음 TBD...**

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

## LLaMA-VID - An Image is Worth 2 Tokens in Large Language Models

- task : 주로 Video-QA

- VLM :  
  - 영화 같은 long video understanding  
  - token 수가 너무 많아서 문제

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/llamaVID/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- architecture  
  - context attention :  
  $$E_t = mean(softmax(Q_t X_t^T) X_t)$$  
  - context token : from video frame and user question
  - content token : from video frame

- contribution  
  - 각 video frame을 두 가지의 token으로 나타냄  
    - context token (one token)  
    - content token (one token으로 compressed될 수도 있고 아닐 수도 있음)
  - hour-long video understanding을 위한 instruction dataset 만듦

## PEEKABOO - Interactive Video Generation via Masked-Diffusion

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/PEEKABOO/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Video Diffusion Model
</div>


- 기존 video generation diffusion model :  
  - 성능 꽤 좋은데 아직 user가 control하기 어려움  
  - 이전에 spatial control을 적용하려면 전체 network를 training시키거나 adapter로 training시키는 과정이 필요했다 
  - 본 논문은 추가적인 training 없이 masked attention module을 사용하여  
  diffusion의 3D UNet을 사용하는 다양한 model에 적용할 수 있는 방법을 제시

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/PEEKABOO/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- contribution :  
  - `UNet` 기반의 video generation model이라면 `spatio-temporal control` 가능  
  (spatio-temporal control : video가 generated될 때 object size, location, and trajectory 등을 user가 control하는 것)
  - `training-free`
  - no additial latency at inference time

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-05-Multimodal/PEEKABOO/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- `Masked attention` :  
  - fg에만 attention하도록 만들기 위해   
  - $$\text{MaskedAttention}(Q, K, V, M) = \text{softmax}(\frac{QK^T}{d}+M)V$$  
  where $$M[i, j] = - \infty$$ if bg(0)  
  where $$M[i, j] = 0$$ if fg(1)  

- binary mask :  
  - image :  
  input BB를 입력으로 받아서 BB object 있는 부분만 fg = 1이 되도록 binary mask $$M_{input}^f[i]$$ 를 만들어서 latent size로 downsample  
  where size of $$n_{frame} \times n_{latents}$$  
  - text :  
  text embedding을 받아서 object 나타내는 단어만 fg = 1이 되도록 mask $$T[j]$$  
  where size of $$n_{text}$$

- `Masked cross attention` :  
  - text와 image 간의 attention
  - cross-attention mask :  
  $$M_{CA}^f[i, j] = fg(M_{input}^f[i]) \ast fg(T[j]) + (1-fg(M_{input}^f[i])) \ast (1-fg(T[j]))$$  
  where $$fg$$ : pixel 또는 text token이 fg이면 1을 반환하고, bg이면 0을 반환  
  where size of $$n_{latents} \times n_{text}$$  
  - cross-attention mask :  
    - image와 text가 둘 다 fg(1)이거나 둘 다 bg(0)이면 1을 반환하고  
    둘 중 하나가 fg(1)이고 둘 중 하나가 bg(0)이면 0을 반환  
    - 즉, `fg와 bg가 서로 attention하지 않도록`!  
    `fg는 fg끼리, bg는 bg끼리 attention하도록`!

- `Masked spatial attention` :  
  - image self-attention for spatial
  - spatial-attention mask :  
  $$M_{SA}^f[i, j] = fg(M_{input}^f[i]) \ast fg(M_{input}^f[j]) + (1-fg(M_{input}^f[i])) \ast (1-fg(M_{input}^f[j]))$$  
  where size of $$n_{latents} \times n_{latents}$$  

- `Masked temporal attention` :  
  - image self-attention for temporal
  - temporal-attention mask :  
  $$M_{TA}^i[f, k] = fg(M_{input}^f[i]) \ast fg(M_{input}^k[i]) + (1-fg(M_{input}^f[i])) \ast (1-fg(M_{input}^k[i]))$$  
  where size of $$n_{frame} \times n_{frame}$$  

- Extension :  
image binary mask를 input BB 받아서 manually 만들지 않고  
text prompt 넣어주면 LLM이 대신 만들어줄 수 있음 (VideoDirectorGPT와 유사)  
$$\rightarrow$$  
그럼 text prompt만 입력으로 넣어주면 user control이 가능한 video를 생성할 수 있음!

## Style Aligned Image Generation via Shared Attention

- CVPR 2024 (oral)

## ControlNet - Adding Conditional Control to Text-to-Image Diffusion Models

## InstructPix2Pix - Learning to Follow Image Editing Instructions
