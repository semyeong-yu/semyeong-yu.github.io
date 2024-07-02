---
layout: distill
title: 3D Rotation-Quaternion
date: 2024-07-01 14:00:00
description: Quaternion for Rotation Matrix
tags: quaternion rotation
categories: 3d-view-synthesis
thumbnail: assets/img/2024-07-01-Quaternion/2.png
giscus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
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

## Lecture 06: 3D Rotations and Complex Representations (CMU 15-462/662)

> referenced video :  
[3D Rotations and Quaternion](https://www.youtube.com/watch?v=YF5ZUlKxSgE&list=PL9_jI1bdZmz2emSh0UQ5iOdT2xRHFHL7E&index=7)  
referenced blog :  
[Quaternion](https://blog.naver.com/hblee4119/223188806834)

## 3D Rotation

- 2D rotation에서는 order of rotations 노상관, but  
3D rotation에서는 `order of rotations 중요`

## Gimbal Lock

- Gimbal Lock :  
Euler angles $$\theta_{x}, \theta_{y}, \theta_{z}$$ 로 회전시킬 때 두 축이 맞물려서 `한 축이 소실`되는 상황  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-01-Quaternion/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    1 -> 2번째 그림 : x축(초록) 회전 / 2 -> 3번째 그림 : z축(파랑) 회전 / 3 -> 4번째 그림 : y축(빨강) 회전
</div>

- 위의 그림에 따르면 Euler angles는 $$x$$(초록), $$y$$(빨강), $$z$$(파랑) 순으로 `상속관계`여서  
$$x$$축(초록)을 회전시키면 그의 자식들인 $$y, z$$축(빨강, 파랑)도 같이 회전한다.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-01-Quaternion/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- 이 때, `Gimbal Lock`은 위의 그림과 같이  
`상속관계에서의 2번째 축(빨강)이 -90도 혹은 90도 회전`했을 때  
`상속관계에서의 1번째 축(초록)과 3번째 축(파랑)이 겹쳐서` 같은 방향으로 회전하기 때문에 발생한다.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-01-Quaternion/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

- 예를 들어, 만약 $$\theta_{y} = \frac{\pi}{2}$$ 로 고정한다면  
$$R_x R_y R_z = {bmatrix} 0 & 0 & 1 \\ sin(\theta_{x}+\theta_{z}) & cos(\theta_{x}+\theta_{z}) & 0 \\ - cos(\theta_{x}+\theta_{z}) & sin(\theta_{x}+\theta_{z}) & 0 \end{bmatrix}$$$$  
이므로 $$\theta_{x}, \theta_{z}$$ 값과 관계없이 `특정 하나의 axis에 대한 회전으로 제약 생겨버림`!  

## Quaternion

- Euler angles vs Quaternion :  
Euler angles는 상속관계이므로 한 번에 계산이 불가능하여 순서대로 회전시켜야 하지만,  
Quaternion은 `한 번에 계산 가능`하여 `동시에 회전`시킬 수 있다!

ddd

- 4 $$\times$$ 1 `quaternion` $$q$$ 으로 3 $$\times$$ 3 `rotation matrix` 만드는 방법 : [build_rotation(r)](https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/utils/general_utils.py#L78)  
```Python
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None] # use normalized quaternion

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
```