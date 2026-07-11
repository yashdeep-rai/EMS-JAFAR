# Local Additions & Progress Assets

This folder contains all custom scripts, checkpoints, and visualization plots created during your local development and used for your project report. These files are not part of the original public JAFAR repository.

---

## Folder Structure

```
local_additions/
├── README.md               # This index file
├── visualize_results.ipynb # Interactive Jupyter Notebook for visual results
├── checkpoints/
│   ├── jafar_upsampler.pth  # Pre-trained JAFAR checkpoint (DinoV2-S)
│   └── jafar_segmentation_probe.pth # Pre-trained VOC segmentation linear probe classifier
└── plots/
    ├── report_visual_img_0.png # Combined qualitative evaluation plot
    ├── original_image.png      # Input image from VOC dataset
    ├── lr_features_pca.png     # Low-resolution feature PCA visualization
    ├── jafar_pca.png           # JAFAR high-resolution feature PCA visualization
    └── overlay_with_prediction.png # Overlaid prediction on top of input image
```

---

## How to Run the Visual Comparison

Open the Jupyter Notebook:
* **[visualize_results.ipynb](file:///home/yashdeep/Documents/JAFAR/local_additions/visualize_results.ipynb)**

Run the cells sequentially. The upsampled feature PCAs and semantic segmentation maps will render directly **inline** in the notebook output, and save copies to `local_additions/plots/`.

---

## Local Benchmarking & Results Summary (For your report)

### 1. Quantitative Probe Evaluation (Pascal VOC 2012)
* **Pixel Accuracy**: **94.91%**
* **Mean IoU (mIoU)**: **89.73%**

### 2. Time & Memory Profiling (GeForce RTX 4060 - 8GB VRAM)
| Resolution | Forward Pass Time | Backward Pass Time | Peak VRAM (Training) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **56 × 56** | 6.61 ms | 21.11 ms | 728.82 MB | **PASSED** |
| **112 × 112** | 6.79 ms | 22.34 ms | 1,092.27 MB | **PASSED** |
| **224 × 224** | 30.09 ms | 83.76 ms | 3,246.14 MB | **PASSED** |
| **448 × 448** | 228.22 ms | — | > 8,192 MB | **OOM** |

*(Note: 448x448 training fails due to the quadratic $O(N^2)$ sequence length memory scaling of attention map computation with respect to image resolution).*
