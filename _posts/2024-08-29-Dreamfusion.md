---
layout: distill
title: DreamFusion
date: 2024-08-29 11:00:00
description: Text-to-3D using 2D Diffusion (ICLR 2023)
tags: sds diffusion nerf 3d rendering 
categories: generative
thumbnail: assets/img/2024-08-29-Dreamfusion/1.png
bibliography: 2024-08-29-Dreamfusion.bib
giscus_comments: false
disqus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
# featured: true
toc:
  - name: Contribution
  - name: Overview
    subsections:
      - name: Random camera, light sampling
      - name: NeRF Rendering with shading
      - name: Diffusion loss with conditioning
      - name: Optimization
  - name: Rendering
    subsections:
      - name: Structure
      - name: Geometry Regularizer
  - name: SDS Loss
  - name: Pseudo Code
  - name: Experiment
  - name: Limitation
  - name: Question
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

## DreamFusion: Text-to-3D using 2D Diffusion (ICLR 2023)

#### Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

> paper :  
[https://arxiv.org/abs/2209.14988](https://arxiv.org/abs/2209.14988)  
project website :  
[https://dreamfusion3d.github.io/](https://dreamfusion3d.github.io/)  
pytorch code :  
[https://github.com/ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)  

## Contribution

- `SDS(Score Distillation) Loss` 처음 제시
- NeRF가 Diffusion(Imagen) model이 내놓을 만한 그럴 듯한 image를 합성하도록 함

## Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Overview  
  - initialize NeRF with random weight
  - for each iter.  
    - camera 위치와 각도, light 위치와 색상을 randomly sampling  
    $$P(camera), P(light)$$    
    - NeRF로 image rendering
    - NeRF param. $$\theta$$ 와 text embedding $$\tau$$ 이용해서 SDS loss 계산
    - update NeRF weight

### Random camera, light sampling

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- `camera` :  
  - 3D model을 `bounded sphere` 내부로 제한하고,  
  spherical coordinate(구 표면)에서 camera 위치를 sampling하여  
  구의 원점을 바라보도록 camera 각도 설정  
  - width(64)에 0.7 ~ 1.35의 상수값을 곱하여 focal length 설정

- `light` :  
  - camera 위치를 중심으로 한 확률분포로부터 light의 위치를 sampling하고  
  (어떤 확률분포 `????`)  
  light 색상도 sampling

### NeRF Rendering with shading

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    albedo : NeRF가 예측한 color
</div>

- rendering 방법 :  
  1. albedo $$\rho$$ 만으로 rendering  
  (기존 NeRF와 동일)  
  2. albedo $$\rho$$ 뿐만 아니라 shading하여 rendering  
  3. albedo $$\rho$$ 를 white $$(1, 1, 1)$$ 로 바꾼 뒤 shading하여 rendering  

- `Shading`의 역할 :  
  - shading 없이 $$\rho$$ 만으로 rendering하면  
  평평한 3D model이 나와도 점수 높게 나옴  
  - shading으로 (빛 반사에 따른) shape 정보까지 고려해서 rendering하면  
  `volume 있는` 3D model이 되도록 촉구

- `NeRF MLP` $$\theta$$ :  
  - MLP output : volume density $$\tau$$ 와 albedo $$\rho$$

- `Normal` $$n$$ :  
$$n = - \frac{\nabla_{\mu} \tau}{\| \nabla_{\mu} \tau \|}$$  
where $$n$$ 은 물체 표면의 법선벡터
  - normal vector의 방향은  
  volume density $$\tau$$ 가 가장 급격하게 변하는 방향, 즉 $$\nabla_{\mu} \tau$$ 의  
  반대 방향

- `Shading` $$s$$ :  
$$s = (l_p \circ \text{max}(0, n \cdot \frac{l - \mu}{\| l - \mu \|})) + l_a $$  
where $$l_p$$ 는 light 좌표 $$l$$ 에서 나오는 light(광원) 색상  
where $$l_a$$ 는 ambient light(환경 조명) 색상  
where $$\mu$$ 는 shading 값을 계산할 surface 위 point 좌표  
where $$\circ$$ 는 element-wise multiplication
  - $$n \cdot (l - \mu)$$ 는 표면에서의 normal vector와 표면에서 광원까지의 vector 간의 내적이며,  
  이는 Lambertian(diffuse) reflectance(난반사)에 의해 광원의 빛이 반사되는 정도를 나타냄  
  왜냐하면, 빛이 표면에 수직으로 들어올수록 많이 반사됨  
  - 만약 빛이 표면 반대쪽에 있어서 또는 back-facing normal로 잘못 예측해서  
  내적 값  $$n \cdot (l - \mu)$$ 이 음수일 경우  
  난반사에 의해 광원의 빛이 반사되는 정도는 0  
  - $$l_p \circ \text{난반사 정도} + l_a$$ 에 의해  
  `광원`의 색상 $$l_p$$ 는 물체 `표면의 난반사 정도에 따라` 반영되고  
  `환경 조명`의 색상 $$l_a$$ 는 물체의 `모든 표면에 일정하게` 반영됨

- `Color` $$c$$ :  
$$c = \rho \circ s$$ 또는 $$c = \rho$$  


### Diffusion loss with conditioning

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- `Latent Diffusion` model :  
  - image $$x$$ 가 아니라 encoder를 거친 image latent vector $$z$$ 에 대해 noising, denoising 수행  
  - noisy $$z_T$$ 와 text embedding vector $$\tau_{\theta}$$ (conditioning)을 concat한 뒤  
  denoising하여 input image와 유사한 확률 분포를 갖도록 학습

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- text embedding vector $$\tau_{\theta}$$ :  
T5-XXL text embedding을 거치기 전에  
text prompt engineering 수행  
  - Elevation angle(고각)이 60도 이상일 때 "overhead view"  
  - azimuth angle(방위각)에 따라 "front view", "side view", "back view"
  - text prompt engineering은 원래 좀 투박하게 하나?

### Optimization

- 실험적인 implementation :  
  - noise level (time) sampling $$t$$ :  
  $$z_t, t \sim U[0, 1]$$ 에서 noise level이 너무 크거나($$t=1$$) 너무 작을 경우($$t=0$$) instability 생기므로  
  noise level $$t \sim U[0.02, 0.98]$$ 로 sampling
  - guidance weight $$w$$ :  
  Imagen이 NeRF에 얼만큼 영향을 미칠지(guide할지)인데,  
  high-quality 3D model을 학습하기 위해서는  
  classifier-free guidance weight $$w$$ 를 큰 값(100)으로 설정  
  (NeRF MLP output color가 sigmoid에 의해 [0, 1]로 bounded되어있으므로 constrained optimization 문제라서 guidance weight 커도 딱히 artifacts 없음)  
  (만약 너무 작은 guidance weight 값을 사용할 경우 object를 표현하는 중간값을 찾고자 하여 over-smoothing됨 `????`)
  - seed :  
  noise level이 높을 때 smoothed density는 distinct modes를 많이 가지지 않고  
  SDS Loss는 mode-seeking property를 가지고 있으므로  
  random seed 바꿔도 실험 결과는 큰 차이 없음

- implementation :  
  - train : TPUv4, 15000 iter., 1.5h with Distributed Shampoo optimizer
  - rendering : 각 cpu는 개별 view를 rendering하는데 사용

## Rendering

### Structure

- Mip-NeRF 360 구조 사용
- entire scene 대신 single object를 generate할 때  
`bounded sphere` 내에서 NeRF view-synthesis 하면 빠르게 수렴 및 좋은 성능
- $$\gamma(d)$$ 를 input으로 받아 배경 색상을 계산하는 별도의 MLP로 `environment map`을 생성한 뒤 그 위에 ray rendering하면 좋은 성능  
  - 배경이 보이는 부분은 배경에서의 누적 $$\alpha$$ 값이 1이도록  
  - 물체 때문에 배경이 안 보이는 부분은 배경에서의 누적 $$\alpha$$ 값이 0이도록

### Geometry Regularizer

- DreamField의 regularization :  
  - `empty space가 불필요하게 채워지는` 것을 방지
  - $$L_T = - \text{min} (\tau, \text{mean}(T(\theta, p)))$$ :  
  평균 `transmittance가 클수록` loss가 작음  
  where $$T(\theta, p)$$ : transmittance with NeRF parameter $$\theta$$ and camera pose $$p$$  
  where $$\tau$$ : 최대값 상수

- Ref-NeRF의 regularization :  
  - normal vector $$n_i$$ 의 back-facing (`물체 안쪽을 향하는`) 문제를 방지  
  - orientation loss $$L = \Sigma_{i} w_i max(0, n_i \cdot d)^2$$ :  
  ray를 쏘면 물체의 앞면만 보이니까  
  물체 표면의 normal vector 방향과 ray 방향의 내적이 음수여야 한다  
  따라서 $$n_i$$ 와 $$d$$ 의 `내적이 양수일 경우` back-facing normal vector이므로 penalize  
    - textureless shading을 쓸 때 해당 regularization이 중요  
    만약 해당 regularization 안 쓰면  
    density field로 구한 normal 방향이 camera 반대쪽을 향하게 되어 shading이 더 어두워짐

## SDS Loss

- NeRF로 rendering한 image $$x$$ 에 noise를 더한 것을 $$z_t$$ 로 두고  
U-Net $$\hat \epsilon_{\phi}(z_t | y, t)$$ 을 빼서 denoising하여 얻은 image의 확률분포가  
2D diffusion prior가 내놓는 image의 확률분포와 비슷하도록 하는 loss이며,  
그 차이만큼 NeRF $$\theta$$ 로 back-propagation 

- 배경지식 :  
  - DDPM Loss : $$E_{t, x_0, \epsilon} [\| \epsilon - \hat \epsilon_{\phi}(\alpha_{t}x_0 + \sigma_{t} \epsilon, t) \|^{2}]$$  
  where $$\epsilon \sim N(0, I)$$  
  where $$\alpha_{t} = \sqrt{\bar \alpha_{t}}$$  
  where $$\sigma_{t} = \sqrt{1-\bar \alpha_{t}}$$  
  - 위의 DDPM Loss는 denoising U-Net param.을 업데이트하기 위함  
  우리는 fixed denoising U-Net을 이용하여  
  NeRF param. $$\theta$$ 를 업데이트하기 위한 SDS Loss 필요!

### Simple Derivation of SDS Loss

- DDPM Loss를 $$\phi$$ 말고 $$\theta$$ 에 대해 미분하고  
constant $$\frac{dz_t}{dx} = \alpha_{t} \boldsymbol I$$ 를 $$w(t)$$ 에 넣으면

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/6.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    x는 NeRF가 생성한 image이고, y는 text embedding vector
</div>

- 위의 U-Net Jacobian은 상당한 연산량을 가지는 데 비해  
작은 noise만 줄 뿐 큰 영향이 없으므로  
SDS Loss에서 U-Net Jacobian term은 생략  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-08-29-Dreamfusion/7.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Derivation of SDS Loss

1. $$\nabla_{\theta} L_{SDS}(\phi, x=g(\theta)) = \nabla_{\theta} E_t[w(t)\frac{\sigma_{t}}{\alpha_{t}}\text{KL}(q(z_t|g(\theta)) \| p_{\phi}(z_t | y, t))]$$ :  
gradient of weighted probability density distillation loss <d-cite key="WaveNet">[1]</d-cite>  

2. $$\text{KL}(q(z_t|g(\theta)) \| p_{\phi}(z_t | y, t)) = E_{\epsilon}[log q(z_t | x = g(\theta)) - log p_{\phi}(z_t | y)]$$  
$$\rightarrow \nabla_{\theta}\text{KL}(q(z_t|g(\theta)) \| p_{\phi}(z_t; y, t)) = E_{\epsilon}[\nabla_{\theta}log q(z_t | x = g(\theta)) - \nabla_{\theta}log p_{\phi}(z_t | y)]$$

3. $$\nabla_{\theta}log q(z_t | x = g(\theta)) = (\frac{d\text{log}q(z_t | x)}{dx} + \frac{d\text{log}q(z_t | x)}{dz_t}\frac{dz_t}{dx})\alpha_{t}\frac{dx}{d\theta} = (\frac{\alpha_{t}}{\sigma_{t}}\epsilon - \frac{\alpha_{t}}{\sigma_{t}}\epsilon)\alpha_{t}\frac{dx}{d\theta}$$ :  
$$q$$ 는 고정된 variance의 noise를 사용하므로 $$\theta$$ 에 대한 forward entropy $$\text{log}q$$ 의 미분 값은 0  
  - $$\frac{d\text{log}q(z_t | x)}{dx}$$ : parameter score function  
  gradient of log probability w.r.t parameters `????` 
  - $$\frac{d\text{log}q(z_t | x)}{dz_t}\frac{dz_t}{dx}$$ : path derivative  
  gradient of log probability w.r.t sample `????`

4. <d-cite key="vargrad">[2]</d-cite>  

mode-seeking property

## Pseudo Code

```Python
params = generator.init() # NeRF param.
opt_state = optimizer.init(params) # optimizer
diffusion_model = diffusion.load_model() # Imagen diffusion model
for iter in iterations:
  t = random.uniform(0., 1.) # noise level (time step)
  alpha_t, sigma_t = diffusion_model.get_coeffs(t) # determine noisy z_t's mean, std.
  eps = random.normal(img_shape) # gaussian noise (epsilon)
  x = generator(params, ...) # NeRF rendered image
  z_t = alpha_t * x + sigma_t * eps # noisy NeRF image
  epshat_t = diffusion_model.epshat(z_t, y, t) # denoising U-Net
  g = grad(weight(t) * dot(stopgradient[epshat_t - eps], x), params) # derivative of SDS loss; stopgradient since do not update diffusion model
  params, opt_state = optimizer.update(g, opt_state) # update NeRF param.
return params
```

## Experiment

TBD

## Limitation

TBD

## Question

TBD