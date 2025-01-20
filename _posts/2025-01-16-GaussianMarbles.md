---
layout: distill
title: Dynamic Gaussian Marbles
date: 2025-01-16 12:00:00
description: Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos (SIGGRAPH 2024)
tags: Gaussian Marble degree freedom dynamic view synthesis
categories: 3d-view-synthesis
thumbnail: assets/img/2025-01-16-GaussianMarbles/1.PNG
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

- Dynamic 4D Gaussian Splatting :  
  - 이전까지는 multiple simultaneous viewpoints of a scene 세팅 (dense multi-camera setup)에서의 recon. 논문들이 많았음  
  $$\rightarrow$$  
  평소의 casual `monocular video`로 4D recon.을 수행해보자!
  - input에 multi-view info.가 없는 underconstrained monocular video더라도  
  prior (`careful optimization strategy` 및 `off-the-shelf depth and motion estimation` 및 `geometry-based regularization`) 이용해서  
  적절한 constraint를 복원할 수 있다! 

- Dynamic Gaussian Marbles :  
monocular setting의 어려움을 해결하기 위해 GS에서 세 가지 사항을 변경  
이를 통해 Gaussian trajectories를 학습할 수 있음
  - isotropic Gaussian Marbles :  
    - `isotropic` Gaussian을 사용함으로써  
    Gaussian의 `degrees of freedom을 줄이고`  
    `local shape보다는 motion과 apperance` 표현하는 데 더 집중하도록 제한
  - hierarchical divide-and-conquer learning strategy :  
    - time 길이가 어느 정도 짧아야 잘 포착할 수 있으므로    
    long video를 short `subsequences로 나누고 optimize by iteratively merging the subsequences`  
    - long-sequence tracking 대신 인접한 subsequences를 붙이는 task로!  
    (guide towards solution with globally coherent motion)  
  - prior :  
  monocular video로도 recon. 잘 수행하기 위해 prior 이용  
    - `image(2D)-space prior` : SAM, CoTracker, DepthAnything
    - `geometry(3D)-space prior` : regularization of Gaussian trajectories with rigidity and Chamfer priors `???`

## Related Work

- Gaussian Splatting :  
TBD

- Dynamic Gaussian Splatting :  
TBD

- Other Dynamic Nerual Scene Representations :  
TBD

## Method

### Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-01-16-GaussianMarbles/1.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-01-16-GaussianMarbles/3.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Dynamic Gaussian Marbles

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-01-16-GaussianMarbles/2.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    simpler Gaussian Marble의 경우에만 generalize well to novel view
</div>

- Gaussian marble :  
  - `isotropic` :  
  $$R = I$$ and $$S = s \in R^{1}$$
    - anisotropic Gaussian은 expensive할 뿐만 아니라  
    underconstrained monocular cam. setting에서는 오히려 degrees of freedom 많으면 poor하다는 걸 실험적으로 발견
  - `semantic instance` :  
  assign each Gaussian marble to semantic instance $$y \in N$$ by SAM-driven TrackAnything
  - `dynamic trajectory` :  
  trajectory $$\Delta X \in R^{T \times 3}$$ : a sequence of translations which maps marble's position change at each timestep

### Divide-and-Conquer Motion Estimation

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-01-16-GaussianMarbles/3.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Training Procedure :  
  - Step 1) `initialization for each frame`  
  initialize Gaussian marbles $$[G_{11}, G_{22}, \ldots, G_{TT}]$$ for each frame  
  (initial marbles $$G_{ii}$$ have trajectory length 1)
    - Step 1-1) obtain prior (`depthmap` and `segmentation`)  
    obtain monocular (LiDAR) depthmap and segmentation from SAM-driven TrackAnything <d-cite key="TrackAnything">[1]</d-cite>
    - Step 1-2) `unproject` from 2D to 3D  
    unproject the depthmap into point cloud  
    perform outlier removal and downsampling
    - Step 1-3) initialize Gaussian marbles and trajectory 
      - Gaussian marbles :  
        - mean $$\mu \in R^{3}$$ : Step 1-2)에서 얻은 pcd
        - color $$c \in R^{3}$$ : pixel color (pixel-aligned Gaussians)
        - `instance class` $$y \in R^{1}$$ : Step 1-1)에서 얻은 segmentation
        - scale $$s \in R^{1}$$ and opacity $$\alpha \in R^{1}$$ : 3DGS 논문에서 했던대로 초기화
      - trajectory :  
        - `trajectory` : $$\Delta X = [\boldsymbol 0] \in R^{T \times 3}$$
  - Step 2) bottom-up divide-and-conquer merge  
  merge short-trajectories into longer trajectories  
  e.g. $$G = [G_{12}, G_{34}, G_{56}, G_{78}] \rightarrow G = [G_{14}, G{58}]$$ 
    - Step 2-1) `motion estimation`  
      - Step 2-1-1) make a pair b.w. adjacent marbles  
        - adjacent Gaussian marble set끼리 a pair로 묶음  
        e.g. $$[(G_{12}^{a}, G_{34}^{b}), (G_{56}^{a}, G_{78}^{b})]$$
        - $$G^{a}$$ 는 merge할 prev. frames' Gaussians이고,  
        $$G^{b}$$ 는 merge할 next frames' Gaussians  
      - Step 2-1-2) $$G^{a}$$ 의 trajectory 확장  
        - goal :  
        $$G_{12}^{a}$$ 의 trajectory인 $$\Delta X = [\Delta X_{1}, \Delta X_{2}]$$ 는 이미 학습되어 merge된 motion이고,  
        $$G_{12}^{a}$$ 의 trajectory와 $$G_{34}^{b}$$ 의 frame $$3$$ 을 잇는 motion $$\Delta X_{3}$$ 을 학습해야 함!  
        - constant-velocity assumption에 따라 trajectory를 $$\Delta X = [\Delta X_{1}, \Delta X_{2}, \Delta X_{3}^{init}]$$ 로 확장
      - Step 2-1-3) trajectory optimization  
        - $$G_{12}^{a}$$ 를 frame $$3$$ 에 render한 뒤  
        $$\Delta X_{3}$$ 이 frame $$3$$ 으로의 motion을 잘 반영하도록 $$\Delta X_{3}$$ 을 업데이트  
        ($$\eta$$ 번 반복 by 아래에서 설명할 Loss)
      - Step 2-1-4) repeat  
        - Step 2-1-2), Step 2-1-3)을 반복  
        until trajectory가 $$G_{34}^{b}$$ 내 모든 frames를 커버할 때까지  
        - e.g. $$G^{a}$$ 의 trajectory를 $$\Delta X = [\Delta X_{1}, \Delta X_{2}, \Delta X_{3}, \Delta X_{4}^{init}]$$ 로 확장한 뒤  
        $$G^{a}$$ 를 frame $$4$$ 에 render한 뒤  
        $$\Delta X_{4}$$ 가 frame $$4$$ 으로의 motion을 잘 반영하도록 $$\Delta X_{4}$$ 을 업데이트  
    - Step 2-2) `merge`  
      - motion estimation을 거치고 나면 $$G_{ij}^{a}$$ 와 $$G_{ij}^{b}$$ 가 같은 frame subsequence $$[i, j]$$ 를 recon.할 것이므로  
      merge by just union $$G_{ij} = G_{ij}^{a} \cup G_{ij}^{b}$$  
      (set size 2배 됨)
      - computational load 줄이기 위해  
      opacity 또는 scale이 너무 작은 Gaussians는 drop하고,  
      random downsampling 수행하여  
      set size를 constant하게 유지
    - Step 2-3) `global adjustment`  
      - merge로 합치고 나서도 still optimized라는 보장이 없기 때문에  
      newly merged Gaussians를 모두 jointly optimize
      - merged set가 $$G_{ij}$$ 라고 했을 때  
      $$[i, j]$$ 내 a frame을 randomly sampling하고  
      $$G_{ij}$$ 의 모든 Gaussians를 해당 frame에 render한 뒤  
      Gaussian $$c, s, \alpha, \Delta X$$ 을 업데이트  
      ($$\beta$$ 번 반복)
      - 그럼 merged Gaussians $$G_{ij}$$ 가 global Gaussians로 인정받을 수 있음!
 
- Inference Procedure :  
  - learned Gaussian trajectories 이용해서  
  render (roll out) into specific timestep $$t$$

### Loss

- Tracking Loss :  
TBD

- Rendering Loss :  
TBD

- Geometry Loss :  
TBD

- 3D Alignment Loss :  
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
Gaussian의 motion을 학습하는 것이라면 없던 object가 등장하거나 원래 있던 object가 frame 밖으로 벗어나는 경우에도 잘 대응할 수 있는지?

- A1 :  
TBD