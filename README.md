# Tools created for Work on Reducing Simulation-to-Reality Gap for Birds-eye-view generation without Ground Truth Supervision

This repository contains helper and analysis scripts developed as part of a Master's thesis on sim-to-real transfer for novel view synthesis in autonomous driving. The work builds on [PixelSplat](https://github.com/dcharatan/pixelsplat), trained on synthetic CARLA/SEED4D data and evaluated zero-shot on real-world nuScenes data.

---

## Overview

The tools are organized around four areas of the research workflow:

**Camera Pose Processing**  
Scripts for transforming nuScenes camera calibrations into the coordinate convention expected by PixelSplat, along with interactive 3D visualization of camera frustums and positions. Outputs include zoomable HTML plots, static PNGs, PLY point clouds, and CSV exports of translations, Euler angles, and quaternions.

**Dataset Preparation**  
Utilities for filtering nuScenes data prior to training and evaluation. This includes brightness-based day/night classification of samples, token-level night scene exclusion for both trainval and test splits, and integration into the dataset loader with summary statistics.

**Image Quality & Domain Gap Analysis**  
Scripts for quantifying the perceptual and distributional gap between synthetic (SEED4D, SecoGAN, Flux2) and real-world (nuScenes) image domains. Metrics include FID, KID, LPIPS, PSNR, and SSIM. Pairwise comparisons are supported across all domain combinations, with consistent sampling logic and optional night-scene exclusion for nuScenes.

**Output Visualization**  
Tools for generating GIF and MP4 visualizations from PixelSplat model outputs. Supports per-scene GIF creation as well as a structured 2-row grid layout showing all six surround cameras alongside POV-rendered, depth-rendered, and BEV-rendered outputs — in both labeled and unlabeled variants.

---

## Dependencies

```bash
pip install nuscenes-devkit pyquaternion plotly matplotlib pillow \
            torch torchvision torch-fidelity lpips opencv-python pandas
```

FFmpeg is required for MP4 export.

---

## Context

All tools were developed in the context of a sim-to-real transfer pipeline for autonomous driving, targeting bird's-eye-view generation from real-world multi-camera input without ground-truth exo-view supervision. The synthetic training data was generated using CARLA via the SEED4D framework and domain-translated using SecoGAN and Flux2. Real-world evaluation was performed on the nuScenes dataset.
