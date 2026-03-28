# 🚁 Game of Drones: AI for Heat Resilience
**🥈 2nd Place Winner | Image-to-Image RGB-to-Thermal Translation Pipeline**

This repository contains our team's 2nd Place winning solution for the AI for Heat Resilience Hackathon. Our goal was to bridge the domain gap between visual (RGB) and thermal spectrums by translating high-resolution drone imagery into accurate, actionable thermal maps for urban environments.

We approached this challenge by iterating through multiple architectures, from state-of-the-art flow-based generative models to custom, multi-source environmental fusion networks. 

**Team:** Galgallo, Adithya, Santosh, Nima

---

## 🏗️ Core Architectures & Approaches
To solve the translation problem with a highly constrained dataset (418 training images), our team developed and evaluated three distinct modeling strategies:

### 1. End-to-End Generative Fine-Tuning
* **Framework:** We utilized `ThermalGen`, a flow-based generative model using Scalable Interpolant Transformers (SiT).
* **Metadata Integration:** We directly fused 64-channel AlphaEarth satellite embeddings into the model via concatenation to anchor the generation process to geospatial context.
* **Training:** The model was fine-tuned end-to-end for 33 epochs.

### 2. U-Net Residual Refinement
* **Strategy:** Treating the baseline generative output as a "rough draft," we built a lightweight U-Net to refine the predictions and fix colormap (Inferno) mapping discrepancies.
* **Inputs:** The network utilized a combination of the RGB drone image, local weather metrics, and AlphaEarth embeddings.
* **Parameters:** Efficient 4.35M parameter residual learning architecture.

### 3. ResNet-Guided Refinement with FiLM
* **Encoder:** A pretrained ResNet18 encoder leveraging ImageNet weights to preserve high-frequency spatial structures (roofs, cars, roads).
* **Decoder:** Custom upsampling blocks with skip-connections to maintain structural integrity.
* **Metadata Injection:** FiLM (Feature-wise Linear Modulation) layers dynamically scale and shift network weights based on a **71-D context vector** (7D local weather metrics + 64D AlphaEarth embeddings).
* **Target Alignment:** Explicitly trained to predict the pure Red channel (Grayscale) to mathematically align with the official evaluation scripts and maximize PSNR/SSIM scores.

---

## 📂 Repository Structure & "Bring Your Own Data" (BYOD)
Due to the proprietary nature of the competition dataset, **no images or metadata are included in this repository**. 

All notebooks in this repository are programmed using dynamic relative paths. To run this code on your local machine, you must clone the repo and recreate this exact directory structure using your own aligned RGB/Thermal data:

```text
your-repo-folder/
│
├── data/                             # ⬅️ CREATE THIS FOLDER & ADD YOUR DATA
│   ├── Train_2/
│   │   ├── RGB/                      # Place training RGB .JPGs here
│   │   └── Thermal/                  # Place training Thermal .JPGs here
│   └── Test_2/
│       └── RGB/                      # Place testing RGB .JPGs here
│
├── alphaearth-emb/                   # Place satellite .tif embeddings here
├── drone_and_weather_metadata.json   # Place weather metadata here
├── code/
│   └── train_test_split.json         # Place split logic here
│
├── ResNet_FiLM_Thermal.ipynb         
├── thermal_refinement_pipeline.py    
└── ... (other notebooks)
```

### Installation
Dependencies are handled automatically within the first cell of the notebooks, including fixes for NumPy 2.x compatibility. Core libraries include: 
`torch`, `torchvision`, `diffusers`, `rasterio`, `torchmetrics`, `lpips`, `timm`, `einops`

---

## 📓 Codebase Guide
We have included our various developmental and final notebooks to demonstrate the team's engineering process.

**Generative Baselines & Fine-Tuning**
* `ThermalGen-XL-2.ipynb`: Zero-shot evaluation of the baseline flow-based generative model. 
* `ThermalGen-L-2-concat.ipynb`: Pipeline for training and concatenating AlphaEarth satellite embeddings into the base model.

**Refinement Networks**
* `thermal_refinement_v2_inferno_fix.ipynb`: The lightweight U-Net refinement architecture, addressing Inferno colormap mismatches.
* `thermal_refinement_v3_resnet34_perceptual.ipynb`: Upgrading the encoder to ResNet34 and introducing LPIPS (VGG) Perceptual Loss to force sharper textures.
* `ResNet_FiLM_Thermal.ipynb`: The ResNet18 architecture utilizing inference caching and FiLM conditioning.

**Automation & Deployment**
* `thermal_refinement_pipeline.py`: A unified Command-Line Interface (CLI) script allowing users to sequentially trigger baseline predictions, refinement training, and final test prediction generation without a Jupyter environment.

---
*Built during the AI for Heat Resilience Hackathon*
