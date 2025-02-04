---
layout: distill
title: NoPoSplat
date: 2025-02-03 10:00:00
description: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (ICLR 2025)
tags: dynamic GS SfMfree
categories: 3d-view-synthesis
thumbnail: assets/img/2025-02-03-NoPoSplat/1.PNG
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  - name: Contribution
  - name: Related Works
  - name: Method
    subsections:
      - name: Architecture
      - name: Gaussian Space
      - name: Camera Intrinsic Embedding
      - name: Training and Inference
  - name: Experiment
    subsections:
      - name: Implementation
      - name: Result
      - name: Ablation Study
  - name: Conclusion
  - name: Question
bibliography: 2025-02-03-NoPoSplat.bib
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

## No Pose, No Problem - Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images

#### Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, Songyou Peng

> paper :  
[https://arxiv.org/abs/2410.24207](https://arxiv.org/abs/2410.24207)  
project website :  
[https://noposplat.github.io/](https://noposplat.github.io/)  
code :  
[https://github.com/cvg/NoPoSplat](https://github.com/cvg/NoPoSplat)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/4.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## Contribution

- model :
  - `unposed` (no extrinsic) `sparse-view` images로부터 3DGS를 통해 3D scene recon.하는 feed-forward network 제시
  - `photometric loss만으로` train 가능  
  (`GT depth 사용 X`, explicit matching loss 사용 X)
  - 본 논문은 intrinsic의 영향을 받는 image appearance에만 의존하여 recon.을 수행하므로  
  `scale ambiguity` 문제 해결을 위해 `intrinsic embedding method` 사용  
  (intrinsic은 input으로 사용)

- downstream tasks :  
  - recon.된 3DGS를 이용하여 novel-view-synthesis 및 pose-estimation task 수행 가능  
    - 특히 limited input image overlap (sparse) 상황에서는 pose-required methods보다 더 좋은 성능
    - 정확히 pose-estimation 수행하는 two-stage coarse-to-fine pipeline 제시
  - generalize well to out-of-distribution data

- Gaussian Space :  
  - `first input view의 local camera coordinate`을 `canonical space`로 고정하고 모든 input view의 3DGS들을 해당 space에서 directly 예측
  - 기존에는 transform-then-fuse pipeline이었는데,  
  본 논문은 global coordinate으로의 `explicit transform 없이` canonical space 내에서의 different views의 fusion 자체를 직접 network로 학습
  - local coordinate에서 global coordinate으로 3DGS를 explicitly transform할 필요가 없으므로  
  explicitly transform하면서 생기는 per-frame Gaussians의 misalignment를 방지할 수 있고, extrinsic pose 없이도 3D recon. 가능

## Related Works

- SfM :  
  - bundle adjustment 등 최적화 과정을 거치는데,  
  off-the-shelf pose estimation method 사용하는 것 자체가 많은 연산을 필요로 하고 runtime 늘림
  - 3D recon.에 only two frames만 input으로 사용하더라도  
  SfM을 통해 해당 two frames의 camera pose를 구하려면 many poses from dense videos 필요 (impractical)
  - textureless area 또는 image가 sparse한 영역에서는 잘 못 함

- Pose-Free Method :  
  - pose-estimation과 3D recon.을 single pipeline으로 통합하자! : <d-cite key="DBARF">[1]</d-cite>, <d-cite key="Flowcam">[2]</d-cite>, <d-cite key="Unifying">[3]</d-cite>
    - pose-estimation과 scene-recon.을 번갈아가며 수행하는 sequential process 에서 error가 쌓이기 때문에  
    SOTA novel-view-synthesis methods보다 성능 bad
  - DUSt3R, MASt3R 계열

- DUSt3R, MASt3R :  
  - 공통점 1)  
  pose-free method  
  - 공통점 2)  
  directly predict in canonical space
  - 차이점 1)  
  DUSt3R, MASt3R는 transformer output이 3D pointmap (point cloud)인데,  
  NoPoSplat은 mean, covariance, opacity, color를 가진 3DGS (rasterization) 사용
  - 차이점 2)  
  NoPoSplat은 DUSt3R, MASt3R 계열과 달리 `GT depth 필요 없고 photometric loss만으로` 훈련 가능 

- pixelSplat, MVSplat :  
  - 차이점 1) (아래 그림 참고)  
  pixelSplat, MVSplat은 먼저 intrinsic을 이용해 2D-to-3D로 unproject(lift)하여 each local-coordinate에서 3DGS를 예측한 뒤 extrinsic을 이용해 world-coordinate으로 transform한 뒤 fuse했는데,  
  NoPoSplat은 canonical space 내에서의 different views의 fusion 자체를 directly network로 학습하기 때문에 `global coordinate으로 transform할 필요가 없으므로` 이에 따른 `misalignment를 방지`할 수 있고 `camera pose (extrinsic)도 필요 없음`
  - 차이점 2)  
  pixelSplat에선 epipolar constraint, MVSplat에선 cost volume이라는 geometry prior를 사용하였는데,  
  NoPoSplat은 (image overlap이 클 때 유리한) `geometry prior들을 사용하지 않음`

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/2.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## Method

### Architecture

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/3.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- I/O :  
$$f_{\theta} : \left\{ (I^{v}, k^{v}) \right\}_{v=1}^{V} \mapsto \left\{ \bigcup (\mu_{j}^{v}, \alpha_{j}^{v}, r_{j}^{v}, s_{j}^{v}, c_{j}^{v}) \right\}_{j=1, \ldots, H \times W}^{v=1, \ldots, V}$$
  - input :  
    - sparse unposed multi-view images $$I$$ (image 개수 $$V$$)
    - camera intrinsics $$k$$ (available from modern devices <d-cite key="intrinsic">[4]</d-cite>)
  - output :  
    - mean $$\mu \in R^{3}$$, opacity $$\alpha \in R$$, rotation $$r \in R^{4}$$, scale $$s \in R^{3}$$, SH $$c \in R^{k}$$ ($$k$$ degrees of freedom)

- Pipeline :  
  - `Encoder, Decoder` :  
    - 특히 input views끼리 content overlap이 적은 상황 (sparse) 에서는  
    epipolar constraint나 cost volume 같은 geometry prior가 없더라도  
    simple ViT 구조만으로도 좋은 성능 달성 가능
    - RGB images를 image tokens로 patchify, flatten한 뒤  
    intrinsic token과 concatenate한 뒤  
    Encoder and Decoder에 feed-forward
  - Gaussian Parameter Prediction Head :  
  DPT 구조
    - `Gaussian Center Head` :  
    Decoder feature 사용
    - `Gaussian Param Head` :  
    RGB image와 Decoder feature 사용  
      - `RGB shortcut` :  
      3D recon.에서 fine texture detail을 잡는 것이 중요하기 때문에 사용
      - Decoder feature :  
      high-level semantic info.

### Gaussian Space

- baseline: `Local-to-Global Gaussian Space`  
  - pixelSplat, MVSplat 등
  - how :  
  먼저 each pixel의 depth를 network로 예측한 뒤  
  predicted depth와 intrinsic을 이용해 2D-to-3D로 unproject(lift)하여 each local-coordinate에서 3DGS 예측한 뒤  
  extrinsic을 이용해 world-coordinate으로 transform한 뒤  
  모든 transformed 3DGS들을 fuse  
  - issue :  
    - local-coordinate에서 world-coordinate으로 transform할 때 `accurate camera pose` (extrinsic) 필요한데, 이는 input view가 sparse한 real-world 상황에서 얻기 어렵
    - 특히 input view가 sparse할 때 또는 out-of-distribution data로 일반화할 때는  
    `each transformed 3DGS들을 조화롭게 combine`하는 게 어렵

- NoPoSplat: `Canonical Gaussian Space`
  - how :  
  first input view를 global referecne coordinate으로 고정한 뒤 ($$[R | t] = [\boldsymbol I | \boldsymbol 0]$$)  
  해당 coordinate 내에서 each input view $$v$$ 마다 set $$\left\{ \mu_{j}^{v \rightarrow 1}, r_{j}^{v \rightarrow 1}, c_{j}^{v \rightarrow 1}, \alpha_{j}, s_{j} \right\}$$ 을 예측  
  - benefit :  
    - global coordinate으로 explicitly transform할 필요가 없으므로 camera pose (extrinsic) 필요 없음
    - explicitly transform-then-fuse하는 게 아니라 fuse 자체를 network로 학습하는 것이기 때문에  
    조화로운 global representation 가능

### Camera Intrinsic Embedding

- Camera Intrinsic Embedding :  
  - issue :  
  only appearance에만 의존하여 3D recon.을 수행함  
  `scale ambiguity` (scale misalignment) 문제 해결 필요!  
  필요한 geometric info.를 제공하기 위해!  
  intrinsic $$k = [f_{x}, f_{y}, c_{x}, c_{y}]$$
  - solve :  
    - Trial 1) Global Intrinsic Embedding by Addition :  
    intrinsic $$k$$ 을 linear layer에 통과시킨 뒤 RGB image token에 add
    - Trial 2) Global Intrinsic Embedding by Concat :  
    intrinsic $$k$$ 을 linear layer에 통과시킨 뒤 RGB image token에 concat
    - Trial 3) Pixel-wise (Dense) Intrinsic Embedding :  
    each pixel $$p_{j}$$에 대해 ray direction $$K^{-1} p_{j}$$ 구한 뒤  
    SH 이용해서 high-dim. feature로 변환한 뒤  
    RGB image와 concat

### Training and Inference

- Loss :  
only photometric loss  
(linear comb. of MSE and LPIPS)

- Relative Pose Estimation :  
  - TBD

pose-estimation coarse-to-fine two-stage pipeline : 처음에 Gaussian center에 PnP algorithm 적용하여 initial rough pose estimate 구한 뒤 photometric loss로 input view와의 alignment를 optimize하면서 pose estimate을 refine

- Evaluation-Time Pose Alignment :  
  - TBD

## Experiment

### Implementation

- Experiment :  
  - Dataset :  
  TBD
  - Metrics :  
  TBD
  - Baseline :  
  TBD
  - Implementation :  
  TBD 

### Result

- Novel View Synthesis :  
TBD

- Relative Pose Estimation :  
TBD

- Geometry Reconstruction :  
TBD

- Cross-Dataset Generalization :  
TBD

- Model Efficiency :  
TBD

- In-the-Wild Unposed Images :  
TBD

### Ablation Study

- Ablation Study :  
  - Output Gaussian Space :  
  TBD
  - Camera Intrinsic Embedding :  
  TBD
  - RGB Shortcut :  
  TBD
  - 3 Input Views instead of 2 :  
  TBD

## Conclusion

- Limitation :  
TBD

## Question

TBD