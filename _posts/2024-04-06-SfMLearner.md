---
layout: distill
title: SfMLearner
date: 2024-04-06 17:00:00
description: Unsupervised Learning of Depth and Ego-Motion from Video
tags: unsupervised depth ego motion video
categories: depth-estimation
thumbnail: assets/img/2024-04-06-SfMLearner/3.png
giscus_comments: true
related_posts: true
toc:
  beginning: true
  sidebar: right
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

# Unsupervised Learning of Depth and Ego-Motion from Video

#### Tinghui Zhou, Matthew Brown, Noah Snavely, David G. Lowe

> paper :  
[https://arxiv.org/abs/1704.07813](https://arxiv.org/abs/1704.07813)  
code :  
[https://github.com/tinghuiz/SfMLearner](https://github.com/tinghuiz/SfMLearner)  

## Introduction

- SfM : Structure from Motion  
end-to-end unsupervised learning from monocular video (only one camera lens)  
- `single-view` depth estimation by per-pixel depth map
- `multi-view` camera motion (= `ego-motion` = `pose`) by `6-DoF transformation matrices`  
- `unsupervised` learning : 직접적인 GT data가 아니라 `view synthesis (reconstruction term)를 supervision`으로 씀

## Related Work

- simultaneous estimation of structure and motion through deep learning
- end-to-end learning of transformation matrix without learning geometry explicitly
- learning of 3D single-view from registered 2D views
- unsupervised/self-supervised learning from video

## Method

#### Approach

Assumption :  
Scenes, which we are interested in, are mostly rigid, so changes across different frames are dominated by camera motion

#### View Synthesis as supervision

- View Synthesis : as supervision of depth and pose (추후 설명 예정)  
- loss function (reconstruction term) :

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

$$p$$ : index of target view's pixel coordinates  
$$s$$ : index of source views  
$$I_{t}(p)$$ : target view  
$$\hat I_{s}(p)$$ : source view warped to target coordinate frame (= reconstructed target view) using predicted depth $$\hat D_{t}$$ and $$4 \times 4$$ camera transformation matrix $$\hat T_{t \rightarrow s}$$ and source view $$I_{s}$$  

- pipeline for depth and pose estimation :

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### Differentiable depth image-based rendering

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

1. `Depth CNN`을 통해 `target view` (single view)로부터 `depth prediction` $$\hat D_{t}$$ 얻기

2. `Pose CNN`을 통해 `target & source view` (multi-view)로부터 $$4 \times 4$$ `camera transformation matrix` $$\hat T_{t \rightarrow s}$$ 얻기

3. target view의 pixels를 source view coordinate으로 `project`하기  
값이 아니라 `대응되는 위치`를 구하기 위해  
projection할 때 depth와 pose 이용  
- monocular camera이므로 두 카메라 사이의 상대적인 위치를 설명하는 $$[R \vert t]$$는 고려 안함
- $$K^{-1}p_{t}$$ : target view coordinate에서 2D 좌표 $$\rightarrow$$ 3D 좌표
- $$\hat D_{t}(p_{t})K^{-1}p_{t}$$ : target view의 3D depth map (= 2D depth $$\times$$ 3D 좌표)  
full 3D volumetric은 아니고, surface만 나타내는 3D target
- $$\hat T_{t \rightarrow s} \hat D_{t}(p_{t})K^{-1}p_{t}$$ : 3D depth map projected from target view to source view
- $$K \hat T_{t \rightarrow s} \hat D_{t}(p_{t})K^{-1}p_{t}$$ : source view coordinate에서 3D 좌표 $$\rightarrow$$ 2D 좌표  
`target view의 pixel 좌푯값을 source view의 좌푯값으로 project하는 데 중간에 depth map이 왜 필요한 거지???`  

4. source view coordinate에서 differentiable bilinear `interpolation`으로 value 얻은 뒤 `warp to target coordinate` (= `reconstructed target view`)  
source view의 pixel 값들을 이용해서 reconstruct target view

#### Modeling the model limitation

Assumption :  
1. objects are static except camera (changes are dominated by camera motion)  
물체들이 움직이지 않아야 Depth CNN과 Pose CNN이 같은 coordinate에 대해 project할 수 있다.

2. there is no occlusion/disocclusion between target view and source view  
target view와 source views 중 하나라도 물체가 가려져서 안보인다면 projection 정보가 없어 학습에 문제가 된다.

3. surface is Lambertain so that photo-consistency error is meaningful  
어떤 방향에서 보든 표면이 isotropic 똑같은 밝기로 보인다고 가정
$$\rightarrow$$ photo-consistency에 차이가 있을 경우 이는 다른 surface를 의미함

#### Overcoming the gradient locality at loss term

1. To improve robustness, train additional network which predicts `explainability soft mask` $$\hat E_{s}$$ (= `per-pixel weight`), and add it to reconstruction loss term.  
deep-learning model은 black-box이므로 explainablity는 중요한 요소

2. trivial sol. $$\hat E_{s} = 0$$을 방지하기 위해, add `regularization` term that encourages nonzero prediction of $$\hat E_{s}$$ 

3. 직접 pixel intensity difference로 reconstruction loss를 얻으므로, GT depth & pose로 project하여 얻은 $$p_{s}$$ 가 low-texture region or far region에 있을 경우 training 방해 (common issue in motion estimation)  
$$\rightarrow$$ 해결 1. use conv. encoder-decoder with small bottleneck  
$$\rightarrow$$ 해결 2. add `multi-scale` and `smoothness loss` term  
(less sensitive to architecture choice, so 이 논문은 해결 2. 적용)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    s : source view image index  /  p : target view pixel index
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    l : multi-scale  /  s : source view image index
</div>


#### Network Architecture

- Network 1. `Single-view Depth CNN`  
input : target view  
output : per-pixel depth map  
DispNet encoder-decoder architecture

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/6.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Network 2. `Multi-view Pose CNN` (아래 figure의 파란 부분)  
input : target view concatenated with all source views  
output : 6-DoF relative poses between target view and each source view  
(Pose CNN estimates `6 channels (3 Euler angles + 3D translation vector)` for each source view, and then it is converted to $$4 \times 4$$ `transformation matrix`)  

`어떻게 transformation matrix로 변환???`

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2024-04-06-SfMLearner/7.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2024-04-06-SfMLearner/7.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- Network 3. `Explainablity soft mask` (= `reconstruction weight per pixel`) (위의 figure의 빨간 부분)  
output : multi-scale explainability masks  
(it estimates `2 channels` for each source view at each prediction layer)  

`weight per pixel인데 왜 2 channels are needed for explainability mask???`  


## Experiments

Train : BN, Adam optimizer, monocular camera (one camera lens), resize input image  
Test : arbitrary input image size  

#### Single-view depth estimation

- train model on the split (exclude frames from test sequences and exclude static scene's pixels with mean optical flow magnitude < 1)  
- pre-trained on Cityscapes dataset / fine-tuned on KITTI dataset / test on Make3D dataset  
- may improve if we also use left-right cycle consistency loss  
- ablation study 결과, explainablity mask를 추가하고 fine-tuning하는 게 더 좋은 성능 도출  

#### Multi-view pose estimation

- trained on KITTI odometry(change in position over time by motion sensor) dataset  
- measurement :  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/8.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

ATE : Absolute Trajectory Error  
left/right turning magnitude : coordinate diff. in the side-direction between start and ending frame at test  
Mean Odom. : mean of car motion for 5-frame snippets from GT odometry dataset  
ORB-SLAM(full) : recover odometry using all frames for loop closure and re-localization  
ORB-SLAM(short) : Ours에서처럼, use 5-frame snippets as input  
$$\rightarrow$$ 특히 small left/right turning magnitude (car is mostly driving forward) 상황에서 Ours가 ORB-SLAM(short)보다 성능 더 좋으므로 monocular SLAM system의 local estimation module을 Ours가 대체할 수 있을 것이라 예상​  
(`SLAM 논문 아직 안 읽어봄. 읽어보자.`)  

#### Visualizing Explainability Prediction

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-06-SfMLearner/9.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    highlighted pixels at explainability mask : predicted to be unexplainable
</div>

explainability = per-pixel weight (confidence 느낌) for reconstruction  

row 1 ~ 3 : due to motion (dynamic objects are unexplainable)  
row 4 ~ 5 : due to occlusion/visibility (disappeared objects are unexplainable)  
row 6 ~ 7 : due to other factors (e.g. depth CNN has low confidence on thin structures)  

## Discussion

#### Contribution

- end-to-end `unsupervised` learning from `monocular` sequences  
(기존에는 gt depth로 depth supervision 또는 calibrated stereo images로 pose supervision이었지만, 본 논문은 `view synthesis (reconstruction)을 supervision으로` 써서 unsupervised learning으로도 comparable performance 달성)  
- depth CNN recognizes common structural features of objects, and pose CNN uses image correspondence with estimating camera motion  

#### Limitation

1. `dynamic objects (X) / occlusion (X) / must be Lambertain surface / vast open scenes (X) / when objects are close to the front of camera (X) / thin structure (X)`  
$$\rightarrow$$ 위의 한계들을 개선하고자 explainablity mask (= per-pixel reconstruction confidence 느낌) 도입했지만, it is implicit consideration

2. `assume that camera intrinsic K is given`, so not generalized to the random videos with unknown camera types

3. predict simplified 3D depth map of `surface` (`not full 3D volumetric representation`)


중간중간에 있는 질문들은 아직 이해하지 못해서 남겨놓은 코멘트입니다.  
추후에 다시 읽어보고 이해했다면 업데이트할 예정입니다.  
혹시 알고 계신 분이 있으면 댓글로 남겨주시면 감사하겠습니다!