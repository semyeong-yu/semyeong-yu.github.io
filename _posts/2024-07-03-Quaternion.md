---
layout: distill
title: Quaternion
date: 2024-07-03 14:00:00
description: Quaternion for Rotation Matrix
tags: quaternion rotation
categories: 3d-view-synthesis
thumbnail: assets/img/2024-07-03-Quaternion/1.png
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

## 3D Gaussian Splatting for Real-Time Radiance Field Rendering

#### Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis

> referenced video :  
[3D Rotations and Quaternion](https://www.youtube.com/watch?v=YF5ZUlKxSgE&list=PL9_jI1bdZmz2emSh0UQ5iOdT2xRHFHL7E&index=7)




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

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-03-Quaternion/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>