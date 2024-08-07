---
layout: post
title: Reconstruction and Synthesis 3D humans in 3D scenes
date: 2024-05-13 14:00:00
description: SoC Colloquium lecture by Siyu Tang (ETH Zurich)
tags: reconstruction synthesis 3D human
categories: 3d-view-synthesis
thumbnail: assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/3.png
giscus_comments: false
disqus_comments: true
related_posts: true
toc:
  beginning: true
  sidebar: right
images:
  compare: true
  slider: true
---

### Reconstruction and Synthesis 3D humans in 3D scenes

#### Introduction to Siyu Tang (ETH Zurich)
 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/14.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### Lecture Summary

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

1. real human : How to reconstruct natural human motions in 3D scenes with a monocular camera?  
2. digital human
3. virtual human  

#### real human

Key : learn motion priors from high quality mocap datasets  

LEMO: Learning Motion Priors for 4D Human Body Capture in 3D Scenes. Zhang. Zhang. Bogo. Pollefeys. Tang. ICCV 2021 (Oral)  
It enforces smoothness in latent space  

1. physics-based prior  
data-based prior  
2. diffusion-based approach : robust for auto-encoder, but optimization may be not that fast  
reinforcement-learning-based approach : policy update can be fast  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### digital human

In Ego-centric motion capture,  
`Collision score guided sampling`  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### virtual human

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Key : Generate 0.25-second(8 frames) Motion Primitives for perpetual motion prediction  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/6.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/8.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/7.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>

#### Both real and virtual human

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/9.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Use both synthetic data and real data for 3D segmentation in Point Clouds  

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/10.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/11.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>


#### Egocentric Synthetic Data Generator

egocentric task : challenging

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/12.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### Future Work

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-05-13-Reconstruction_Synthesis_3D_humans/13.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
