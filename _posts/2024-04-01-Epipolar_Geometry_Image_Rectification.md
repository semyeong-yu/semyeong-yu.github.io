---
layout: post
title: Epipolar Geometry & Image Rectification
date: 2024-04-01 17:00:00
description: Epipolar Geometry & Image Rectification
tags: epipolar fundamental essential image rectification
categories: depth-estimation
thumbnail: assets/img/2024-04-01-Epipolar_Geometry_Image_Rectification/1.png
giscus_comments: true
related_posts: true
toc:
  beginning: true
  sidebar: right
images:
  compare: true
  slider: true
featured: true
---

## Epipolar Geometry

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-01-Epipolar_Geometry_Image_Rectification/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-01-Epipolar_Geometry_Image_Rectification/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### image plane / epipolar plane / baseline / epipole / epipolar line  

X : 3D point  
$$x_L, x_R$$ : projected 2D point in left and right image  
파란색 면 : image plane  
초록색 면 : `epipolar plane`  
$$O_L, O_R$$ : center of left and right camera  
직선 $$O_L O_R$$ : `baseline`  
epipolar pencil : set of epipolar planes  
`epipole` : intersection of baseline and image plane  
$$e_L, e_R$$ : epipole of left and right camera  
`epipolar line` :
- intersection of image plane and epipolar plane
- 빨간 선 $$l_R$$ : 직선 $$x_R e_R$$ (projected 2D point와 epipole은 epipolar line 위에 있다)
- left image plane 위의 같은 점 $$x_L$$로 project 되는 모든 3D points $$X, X_1, X_2, \cdots$$를 right image plane에 project했을 때 그려지는 선

#### normalized coordinates, pixel coordinates / intrinsic, extrinsic parameters / homography matrix / projection matrix

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-04-01-Epipolar_Geometry_Image_Rectification/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

real-world의 3D 좌표 (X, Y, Z) 에 있는 물체를 카메라에 투영하기 위해 Z (= 깊이) 값을 1로 정규화한 평면을 `normalized plane`이라 하고, $$(\frac{X}{Z}, \frac{Y}{Z}, 1)$$의 좌표값을 갖는다. 이를 image로서 나타내기 위해 초점거리를 곱해주고 원점을 정중앙에서 좌상단으로 바꿔서 normalized plane 상의 normalized coordinates $$(\frac{X}{Z}, \frac{Y}{Z}, 1)$$을 `image plane 상의 pixel coordinates` $$(\frac{X}{Z} \ast f_x - \frac{W}{2}, \frac{Y}{Z} \ast f_y - \frac{H}{2}, 1)$$ 로 변환할 수 있는데, 이 때 곱하게 되는 행렬이 바로 intrinsic matrix (= calibration matrix) K 이다. 그리고 이렇게 intrinsic parameters를 구하는 과정을 Camera Calibration 이라 부른다.  

그런데, 카메라의 각도 혹은 위치가 달라지면 맺히는 이미지 자체도 달라지기 때문에 intrinsic matrix를 곱하기 전에 camera의 rotation 및 translation을 먼저 고려해주어야 하는데, 이 때 곱하게 되는 행렬이 바로 extrinsic marix $$[R \vert t]$$ 이다.  

- `K : intrinsic parameters` (3x3 calibration matrix) (초점거리 곱하고 원점 바꾸는 등 카메라 자체의 특성)

- `R, t : extrinsic parameters` (3x3 rotation, 3x1 translation matrix) (두 카메라의 상대적인 위치, 각도)  

즉, 정리하면 3D point [X, Y, Z]가 image plane [x, y]에 맺히는 projection 과정은 아래의 수식을 따른다.

$$\begin{bmatrix} x \\ y \\ z \end{bmatrix}$$ = $$\begin{bmatrix} f_x & 0 & -W/2 \\ 0 & f_y & -H/2 \\ 0 & 0 & 1 \end{bmatrix}$$ $$[R \vert t]$$ $$\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$  

$$ x_L \Leftrightarrow x_R $$ : homography matrix H (projection of 2D point to 2D point) (`$$x_R = H x_L$$`)  
$$ X \Leftrightarrow x_L $$ : projection matrix $$P_L$$ (`$$x_L = P_LX$$  where  $$P_L = K_L[R \vert t]$$`)

#### Fundamental matrix

1. $$x_R = H x_L$$
2. $$e_R$$과 $$x_R$$은 직선 $$l_R$$ 위에 있으므로 `$$e_{R}^{T} l_R = 0$$ and $$x_{R}^{T} l_R = 0$$`  
예를 들어, 직선 2x+y-2z = 0에 대해 $$l_R$$은 (2, 1, -2)이고, $$e_R$$ 및 $$x_R$$은 직선 위에 있는 점 (x, y, z)이다.
3. 위의 1.과 2.로부터  `$$l_R = e_R \circledast x_R` = e_R \circledast H x_L = F x_L$$   where  F = fundamental matrix = $$e_R \circledast H$$
4. 위의 2.와 3.으로부터 $$x_{R}^{T} l_R = x_{R}^{T} F x_L = 0$$
5. 위의 2.와 3.으로부터 $$e_{R}^{T} l_R = e_{R}^{T} F x_L = 0$$ 이고, 모든 $$x_L$$에 대해 $$e_{R}^{T} F = 0$$을 만족하므로 $$e_R$$은 F의 left null vector이다. (유사한 방법으로 $$e_L$$은 F의 right null vector이다.)  

즉, fundamental matrix와 관련된 식을 정리하면
- `fundamental matrix : $$F = e_R \circledast H$$`
- `correspondence condition : $$x_{R}^{T} F x_L = 0$$`
- `epipolar line : $$l_R = F x_L$$`
- `epipole : $$e_{R}^{T} F = 0$$` ($$e_R$$은 F의 left null vector)

#### Essential matrix

essential matrix는 fundamental matrix의 specialization으로, pixel coordinates이 특별히 `calibrated camera들을 다루는 normalized image coordinates (K = I)인 경우`에 사용된다. 즉, K = I 여서 $$x^{\ast} = PX = K[R \vert t]X = [R \vert t]X$$ 를 만족하는 $$x^{\ast}$$을 normalized coordinates에 있는 image point라 부른다.  

그리고 epipolar constraint란, vector $$x_L O_L$$과 vector $$x_R O_R$$과 vector $$O_L O_R$$이 같은 평면 epipolar plane 위에 있다는 것이다. 이를 normalized coordinates에서 생각하면, `$$x_R^{\ast}$$과 $$Rx_L^{\ast}$$과 t가 같은 평면 epipolar plane 위에 있다`는 뜻이므로 (그 이유는 아래의 Algebraic derivation을 참고하자) 이를 간단하게 수식으로 표현하면 `$$x_R^{\ast T}(t \circledast Rx_L^{\ast}) = 0$$` 이다.  

즉, essential matrix와 관련된 식을 정리하면
- `essential matrix : $$E = t \circledast R$$`
- `correspondence condition : $$x_R^{\ast T} E x_L^{\ast} = 0$$`  

#### ​Relationship between fundamental matrix and essential matrix

