---
layout: distill
title: Dynamic Gaussian Marbles
date: 2025-01-16 12:00:00
description: Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos (SIGGRAPH 2024)
tags: Gaussian Marble degree freedom dynamic view synthesis
categories: 3d-view-synthesis
thumbnail: assets/img/2025-01-16-GaussianMarbles/1m.PNG
giscus_comments: false
disqus_comments: true
related_posts: true
bibliography: 2025-01-16-GaussianMarbles.bib
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

## Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos

#### Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost, Federico Tombari, Gordon Wetzstein, Leonidas Guibas

> paper :  
[https://arxiv.org/abs/2406.18717](https://arxiv.org/abs/2406.18717)  
project website :  
[https://geometry.stanford.edu/projects/dynamic-gaussian-marbles.github.io/](https://geometry.stanford.edu/projects/dynamic-gaussian-marbles.github.io/)  

## Contribution

- 기존 dynamic GS 기법들은 dense multi-view video (controlled capture setting)를 supervision으로 필요로 했는데  
본 논문은 `평소 casually captured monocular video만으로` dynamic GS 수행!  

- `Dynamic Gaussian Marbles` :  
monocular setting의 어려움을 해결하기 위해 GS에서 세 가지 사항을 변경  
이를 통해 Gaussian trajectories를 학습할 수 있음
  - isotropic Gaussian Marbles :  
  Gaussian의 `degrees of freedom을 줄이고`  
  `local shape보다는 motion과 apperance` 표현하는 데 더 집중하도록 제한
  - hierarchical divide-and-conquer learning strategy :  
  guide towards solution with globally coherent motion
  - prior :  
  point tracking에 이점 있는  
  `tracking loss`를 통해 `image-level and geometry-level prior`를 추가  

## Introduction

- challenging monocular video :  


## Related Work

TBD

## Method

### Dynamic Gaussian Marbles

TBD

### Divide-and-Conquer Motion Estimation

TBD

### Loss

TBD

## Experiment

### Dataset

TBD

### Implementation

TBD

### Results

TBD

### Ablation Study

TBD

## Conclusion

- Limitation :  
  - TBD

## Question

- Q1 :  
TBD

- A1 :  
TBD