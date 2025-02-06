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

`pose-free generalizable sparse-view 3D recon. model in canonical Gaussian space!`

- model :
  - `unposed` (no extrinsic) `sparse-view` images로부터 3DGS를 통해 3D scene recon.하는 feed-forward network 제시
  - `photometric loss만으로` train 가능  
  (`GT depth 사용 X`, explicit matching loss 사용 X)
  - 본 논문은 intrinsic의 영향을 받는 image appearance에만 의존하여 recon.을 수행하므로  
  `scale ambiguity` 문제 해결을 위해 `intrinsic embedding method` 사용  
  (intrinsic은 input으로 사용)
  - covariance, opacity, color를 예측하는 Gaussian Param. Head에서 fine texture detail 주기 위해 `RGB shortcut` 사용

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
  explicitly transform하면서 생기는 per-frame Gaussians의 misalignment를 방지할 수 있고, extrinsic pose 없이도 (pose-free) 3D recon. 가능

## Related Works

- SfM :  
  - bundle adjustment 등 최적화 과정을 거치는데,  
  off-the-shelf pose estimation method 사용하는 것 자체가 많은 연산을 필요로 하고 runtime 늘림
  - 3D recon.에 only two frames만 input으로 사용하더라도  
  SfM을 통해 해당 two frames의 camera pose를 구하려면 many poses from dense videos 필요 (impractical)
  - textureless area (원형 호수 등) 또는 image가 sparse한 영역에서는 부정확한 pose 내놓음

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
  pixelSplat에선 epipolar constraint, MVSplat에선 cost volume이라는 geometry prior를 사용하였는데  
  image view overlap이 적을 때는 geometry prior가 정확하지 않음.  
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
  where view $$1$$ : canonical Gaussian space
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
canonical space에 3DGS들이 있다는 전제 하에  
`two-stage coarse-to-fine pipeline`
  - Coarse Stage :  
  Gaussian center에 `PnP algorithm with RANSAC` (efficient as done in ms) 적용하여  
  `initial rough pose estimate` 구하기
  - Fine Stage :  
  `3DGS param.을 freeze`한 채  
  training에 사용했던 `photometric loss`를 이용해  
  target view와 align되도록 rough `target camera pose를 optimize`(refine)
    - automatic diff.에서의 overhead를 줄이기 위해  
    camera Jacobian을 계산 <d-cite key="GSslam">[5]</d-cite>

- Evaluation-Time Pose Alignment :  
  - unposed input images의 경우  
  scene은 다른데 rendered two images는 같을 수 있으므로  
  just two input views로 3D scene recon. 수행하는 건 사실 ambiguous
  - GT camera pose를 이용하는 other baseline들 <d-cite key="pose1">[6]</d-cite>, [7](https://semyeong-yu.github.io/blog/2024/pixelSplat/)과 비교하기 위해 (evaluation purpose)  
  pose-free methods <d-cite key="nopose1">[8]</d-cite>, <d-cite key="nopose2">[9]</d-cite>의 경우 target view에 대한 camera pose를 optimize한 뒤 비교에 사용  

## Experiment

### Implementation

- Experiment :  
  - Dataset :  
    - training :  
    RE10K (RealEstate10k) : indoor real estate  
    DL3DV : outdoor (camera motion pattern 더 다양)
    - zero-shot generalization :  
    ACID : nature scene by drone  
    DTU  
    ScanNet  
    ScanNet++  
    in-the-whild mobile phone capture  
    SORA-generated images
  - camera overlap :  
  SOTA dense feature matching method <d-cite key="ROMA">[10]</d-cite> 로  
  input images' camera overlap 정도를 측정하여  
  small (0.05%-0.3%), medium (0.3%-0.55%), large (0.55%-0.8%)로 나눔
  - Baseline :  
    - pose-required novel-view-synthesis :  
    pixelNeRF, AttnRend, pixelSplat, MVSplat
    - pose-free novel-view-synthesis and relative pose estimation :  
    DUSt3R, MASt3R, Splatt3R, CoPoNeRF, RoMa
  - Implementation :  
  encoder, decoder, Gaussian center head는 MASt3R의 weights로 initialize하고  
  (사실 scratch부터 training해도 성능 비슷하긴 함)  
  Gaussian param head는 randomly initialize  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/5.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Result

- Novel View Synthesis :  
  - SOTA pose-free (DUSt3R, MASt3R, Splatt3R) :  
    - DUSt3R 계열은 `per-pixel depth loss`에 의존하기 때문에 each views를 `fuse하는 게 어렵`  
    그래서 대부분 상황에서 NoPoSplat이 훨씬 더 좋음
  - SOTA pose-required (pixelSplat, MVSplat) :  
    - pixelSplat, MVSplat은 `input view overlap이 작을 때 부정확한 geometry prior` (epipolar constraint, cost volume)을 사용하기 때문에  
    image view overlap이 작은 상황에서는 NoPoSplat이 더 좋음  
    - pixelSplat, MVSplat은 `transform-then-fuse strategy`를 사용하는데 `misalignment`로 부정확할 수 있기 때문에  
    canonical space에서 directly 예측하는 NoPoSplat이 더 좋을 수 있음

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/6.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/7.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Relative Pose Estimation :  
    
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/8.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Geometry Reconstruction :  
  - SOTA pose-required (pixelSplat, MVSplat) :  
    - pixelSplat, MVSplat은 explicitly transform-then-fuse하는 과정에서 두 input images의 경계 영역에서 misalignment (아래 그림에서 파란색 화살표로 표기) 가 있고,  
    input views' overlap이 적을 때는 geometry prior가 부정확해서 distortion (아래 그림에서 분홍색 화살표로 표기) 있는데,  
    NoPoSplat은 canonical space에서 directly 예측하므로 해결

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/9.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Cross-Dataset Generalization :  
NoPoSplat은 geometry prior를 사용하지 않으므로 다양한 scene type에 adapt 가능  
심지어 ScanNet++로의 zero-shot generalization에 대해 RE10K로 훈련시킨 NoPoSplat과 ScanNet++로 훈련시킨 pose-required Splatt3R을 비교했을 때 NoPoSplat이 더 좋음!

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/10.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Model Efficiency :  
NoPoSplat은 0.015초만에 (66 FPS) 3DGS 예측 가능  
(additional geometry prior 안 쓰니까 speed 빠름!)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/11.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
  inference on RTX 4090 GPU
</div>

- In-the-Wild Unposed Images :  
3D Generation task에 적용 가능!  
먼저 text/image to multi-image/video model 이용해서 sparse scene-level multi-view images 얻은 뒤  
Ours (NoPoSplat) 이용해서 3D model 얻음

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/13.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Ablation Study

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-02-03-NoPoSplat/12.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- Ablation Study :  
  - `Output Canonical Gaussian Space` :  
  transform-then-fuse pipeline of pose-required methods has `ghosting artifacts`
  - `Camera Intrinsic Embedding` :  
  no intrinsic leads to `blurry` results due to `scale ambiguity`  
  실험적으로 intrinsic token concat. 방식이 best
  - `RGB Shortcut` :  
  no RGB Shortcut leads to `blurry` results in texture-rich areas  
  (위 그림의 quilt in row 1 and chair in row 3)
  - `3 Input Views` instead of 2 :  
  baselines과의 공평한 비교를 위해 NoPoSplat은 two input-views setting을 사용했는데  
  three input-views를 사용할 경우 성능이 훨씬 좋아졌음!

## Conclusion

- Future Work :  
NoPoSplat은 static scene에만 적용했는데, dynamic scene에 NoPoSplat의 pipeline을 확장 적용!

- Limitation :  
  - `camera intrinsic은 known`이라는 걸 가정!
  - feed-forward model은 `non-generative`하므로 `unseen region`에는 대응 못 함
  - `static scene`에 적용

## Question

- Q1 :  
사실 NoPoSplat은 camera pose 이용한 global coordinate으로의 explicit transform이나 geometry prior (epopiolar constraint, cost volume 등)나 GT depth 없이  
오로지 implicit network의 학습에 의존하여 scene recon. 능력을 학습하겠다는 건데  
photometric loss만으로도 잘 학습이 되나? two input images 경계면의 smoothness 등 추가 regularization loss 추가해주는 게 낫지 않음?

- A1 :  
TBD

- Q2 :  
photometric loss에만 의존하기 때문에 ViT semantic info. 말고도 more info. 주기 위해 intrinsic과 RGB shortcut을 사용하는데  
둘 말고 또 추가하면 좋은 거 있을까?

- A2 :  
TBD