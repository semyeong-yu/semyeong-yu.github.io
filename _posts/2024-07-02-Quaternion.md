---
layout: distill
title: 3D Rotation-Quaternion
date: 2024-07-02 14:00:00
description: Quaternion for Rotation Matrix
tags: quaternion rotation
categories: 3d-view-synthesis
thumbnail: assets/img/2024-07-02-Quaternion/1.png
giscus_comments: true
related_posts: true
# toc:
#   beginning: true
#   sidebar: right
featured: true
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

#### 3D Rotation

- Degrees of Freedom = 3 

- 2D rotation에서는 order of rotations 노상관, but  
3D rotation에서는 order of roations 중요

- 


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