---
layout: distill
title: PhysGaussian
date: 2024-12-21 12:00:00
description: Physics-Integrated 3D Gaussians for Generative Dynamics (CVPR 2024)
tags: 3DGS multi GPU parallel
categories: 3d-view-synthesis
thumbnail: assets/img/2024-12-21-PhysGaussian/1.png
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  - name: Contribution
  - name: Overview
  - name: Backgrounds on Continuum Mechanics
  - name: MPM
  - name: Physics-Integrated 3DGS
  - name: Orientation of SH
  - name: Incremental Evolution of Gaussians
  - name: Internal Filling
  - name: Anisotropy Regularizer
  - name: Experiments
  - name: Limitation

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

## PhysGaussian - Physics-Integrated 3D Gaussians for Generative Dynamics

#### Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, Chenfanfu Jiang

> paper :  
[https://arxiv.org/abs/2311.12198](https://arxiv.org/abs/2311.12198)  
project website :  
[https://xpandora.github.io/PhysGaussian/](https://xpandora.github.io/PhysGaussian/)  
code :  
[https://github.com/XPandora/PhysGaussian](https://github.com/XPandora/PhysGaussian)  
reference :  
[https://xoft.tistory.com/101](https://xoft.tistory.com/101)

## Contribution

- `Physics Simulation`을 `3DGS`에 결합 :  
3DGS에 부피, 질량, 속도 (Physics) property를 부여하여  
3DGS가 시간에 따라 물리 법칙에 따라 변화

## Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-12-21-PhysGaussian/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Step 1) `3DGS optimization`  
Anisotropic Loss를 추가하여 3DGS를 둥글둥글하게 만듦
- Step 2) `3DGS Internel Filling`  
Object 내부 공간을 3DGS로 채워서 continuum(연속체)로 만듦
- Step 3) `Physics Integration`  
  - Dynamics :  
  3DGS에 부피, 질량 부여하여  
  시간에 따라 Continuum Mechanics(연속체 역학)이라는 물리 법칙 따르도록 함  
  - Kinematics :  
  Gaussian Evolution, SH Transform을 통해  
  시간에 따른 물리적인 변화를 3DGS로 모델링

## Backgrounds on Continuum Mechanics

- Conservation of Mass (질량 보존 법칙) :  
시간 $$t$$ 가 바뀌어도 infinitesimal region 내 질량은 항상 일정하게 유지된다!!  
$$\int_{B_{\epsilon}^{t}} \rho (x, t) = \int_{B_{\epsilon}^{0}} \rho (\phi^{-1}(x, t), 0)$$  
  - $$B_{\epsilon}^{t}$$ : infinitesimal region at $$t$$
  - $$\rho(x, t)$$ : density field at $$x, t$$  
  - $$x = \phi(x_{0}, t)$$ : deformation map from $$x_{0}, 0$$ to $$x, t$$

- Conservation of Momentum (운동량 보존 법칙) :  
시간 $$t$$ 가 바뀌어도 물질의 운동량은 변하지 않는다!!  
운동량 변화량 : $$\rho(x, t) \overset{\cdot}{v}(x, t) = \nabla \cdot \sigma(x, t) + f^{ext}$$  
  - $$\overset{\cdot}{v}(x, t)$$ : 가속도 field at $$x, t$$  
  - $$\sigma = \frac{1}{det(F)}\frac{\partial \psi}{\partial F}F^{E}(F^{E})^{T}$$ : Cauchy stress tensor (물체 내부에서 발생하는 응력)  
  where $$\psi(F)$$ : hyperelastic energy density  
  where deformation field gradient $$F = F^{E} F^{P}$$  
    - $$F^{E}$$ : elastic part (탄성)  
    물체에 stress를 가해서 조직에 구조적인 변형이 발생한 후,  
    stress를 제거했을 때 원래 상태로 되돌아가는 성질  
    - $$F^{P}$$ : plastic part (소성)  
    물체에 stress를 가해서 조직에 구조적인 변형이 발생한 후,  
    stress가 탄성 범위를 넘어가서  
    stress를 제거하더라도 원래 상태로 되돌아오지 않는 성질
  - $$f^{ext}$$ : external force per unit volume

## MPM

- 핵심 :  
`particle과 grid 간에 운동량이 상호작용`하여  
이 과정에서 질량과 운동량이 보존되어  
`simulation` 했을 때 현실과 비슷
  - Lagrangian Particle Domain :  
  particle의 위치, 질량, 운동량, 응력, 부피, 외력 등을 모델링하고  
  particle 별로 추적하여 update
  - Eulerian Grid Domain :  
  공간을 3D grid로 나누어서 grid를 통해 particle 이동  
  cell의 크기가 작아질수록 정확도는 올라가지만 연산속도는 느려짐

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-12-21-PhysGaussian/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Procedure :  
  - Particle to Node :  
  each particle의 물리량을 grid 상의 adjacent 8 nodes로 분배
  - Nodal Solution :  
  집계된 힘을 이용해서 each node의 $$a = \frac{F}{m}$$, $$v$$ 를 update
  - Node to Particle :  
  each node의 $$a, v$$ 를 particle로 전파 by weighted sum
  - Update Particles :  
  each particle의 $$a, v$$ 이용해서 새로운 particle 위치 갱신

실험 예시 : [SIGGRAPH2018](https://vimeo.com/267058393)

## Physics-Integrated 3DGS

그렇다면 어떻게 물리 법칙을 3DGS에 적용할까??

- local affine transformation of deformation map $$\phi$$ :  
$$\tilde \phi (X, t) = x_{p} + F_{p} (X - X_{p})$$
  - $$X$$ : arbitrary point
  - $$X_{p}$$ : particle $$p$$ 의 initial point
  - $$x_{p}$$ : particle $$p$$ 의 current point
  - $$F_{p}$$ : 점이 어떻게 이동하는지에 대한 deformation gradient matrix  
  (물리 법칙 적용)

- Physics-Integrated 3DGS :  
  - 3DGS position, covariance matrix 변화 :  
  $$x_{p}(t) = \tilde \phi (X_{p}, t)$$  
  $$\Sigma_{p}(t) = F_{p}(t) \Sigma_{p} F_{p}(t)^{T}$$
  - Gaussian 수식 변화 :  
  ddd

## Orientation of SH
## Incremental Evolution of Gaussians
## Internal Filling
## Anisotropy Regularizer
## Experiments
## Limitation