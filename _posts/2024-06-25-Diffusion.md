---
layout: distill
title: Diffusion
date: 2024-06-25 15:00:00
description: Diffusion Study
tags: diffusion generative
categories: generative
thumbnail: assets/img/2024-06-25-Diffusion/1.png
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

## Diffusion

#### Diffusion Model

- forward process : `fixed` Gaussian noise 더함
- reverse process : `learned` Gaussian noise 뺌 (mean, std를 학습)

#### Likelihood

아래 둘 다 관측값 $$x$$가 나올 확률인데,
- `probability` $$P(x | \theta)$$ : `확률 분포가 고정`된 상태에서, `관측되는 사건이 변화`할 때의 확률  
예: 선택 가능한 정수를 1~5로 제한(확률 분포 고정)했을 때, 관측 목표값이 1~5 중 한 개의 숫자(관측 사건 변화)일 경우
- `likelihood` $$L(\theta | x)$$ : `관측된 사건이 고정`된 상태에서, `확률 분포 몰라서 가정`할 때의 확률  
예: 선택 가능한 정수를 1~5가 아니라 1~10 또는 4~50으로 바꾸면서(확률 분포 모름), 2가 관측될 확률을 계산(관측 사건 고정)할 경우  
예: 어떤(모르는) 확률 분포를 따르는 task를 n회 반복 수행하여 관측했을 때 random var. 종류를 가정할 수도 있고 특정 random var.의 parameter를 가정할 수도 있다  

#### Markov process

- `Markov` process (= Markov chain = `memoryless` process) : Markov property를 가지는 discrete stochastic process  
$$P[s_{t+1}|s_t] = P[s_{t+1}|s_1, \ldot, s_t]$$

#### KL-divergence

$$H(p, q) = - \sum p_i log q_i$$ : 두 확률분포 p, q의 cross entropy (보통 $$p$$는 GT, $$q$$는 predicted)  
$$H(p) = - \sum p_i log p_i$$ : p's entropy (상수값)  
$$KL(p \| q) = H(p, q) - H(p) = \sum p_i log \frac{p_i}{q_i}$$ : 두 확률분포 p, q의 차이  
$$H(p)$$는 상수값이므로 KL-divergence minimize $$\eq$$ cross entropy minimize  
$$KL(p \| q) \simeq \frac{1}{N} \sum_{n=1}^{N} {-log q(x_n | \theta) + log p(x_n)}$$ : $$log p(x_n)$$은 $$\theta$$에 독립이므로 KL-divergence minimize $$\eq$$ (negative log) likelihood maximize  

KL-diverence 특성 :  
1. $$KL(p \| q) \geq 0$$ : p = q일 때 최소
2. $$KL(p \| q) \neq KL(q \| p)$$ (asymmetric) : 거리 개념이 아님  
거리 개념으로 쓰는 방법 : 2가지 KL-divergence를 평균내는 방식의 $$JSD(p \| q)$$  

#### Diffusion Algorithm

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-25-Diffusion/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

> 출처 블로그 :  
[Diffusion Model](https://xoft.tistory.com/32)  
[DDPM 수식 유도](https://xoft.tistory.com/33?category=1156151)  