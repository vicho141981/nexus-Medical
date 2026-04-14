# NEXUS Medical

**Physics-Informed Neural Networks for Brain Tumor Analysis**

NEXUS Medical is a desktop application that estimates the biological invasiveness index (D/ρ) of brain tumors from MRI segmentation data using Physics-Informed Neural Networks (PINNs) and the Fisher-KPP reaction-diffusion equation.

---

## What NEXUS does

NEXUS takes a BraTS MRI case, extracts the tumor segmentation mask, and fits a PINN to the Fisher-KPP PDE:

```
∂u/∂t = D·∇²u + ρ·u·(1 - u)
```

From this fit it estimates:
- **D** — tumor diffusion coefficient (cm²/month)
- **ρ** — tumor proliferation rate (1/month)  
- **D/ρ** — invasiveness index (primary clinical observable)
- **v = 2·√(D·ρ)** — tumor wave front speed (cm/month)

The D/ρ index is used to stratify clinical risk:

| D/ρ | Risk level | Recommendation |
|-----|-----------|----------------|
| ≥ 0.18 | HIGH | Consider wide resection |
| ≥ 0.043 | MEDIUM | Frequent follow-up |
| < 0.043 | LOW | Standard protocol |

---

## Architecture

```
nexus_ui.py              — PySide6 desktop UI + VTK 3D viewer
nexus_report.py          — Clinical PDF report generator (ReportLab)
nexus_brats_pipeline_v4.py  — BraTS batch pipeline (20 cases)
nexus_validacion_v7.py   — Scientific validation vs Zhang et al. 2024
nexus_viz_v1..v4.py      — Standalone VTK visualizers
train_sensor_fusion_v2.py   — MultiheadAttention sensor fusion module
```

### PINN architecture
- **Fourier Feature Encoder (FE2D):** Bx=(1,32), By=(1,32), Bt=(1,16) → 160 features
- **TNet:** Linear(160,256) → Tanh × 4 → Linear(64,1) → Sigmoid
- **Optimizer:** Adam (lr=1e-3 network, lr=5e-3 parameters) + CosineAnnealingLR
- **Training:** Stage 1 (epochs 1-2000): IC + image loss only. Stage 2 (epochs 2001+): PDE loss activated with progressive ramping

---

## Hardware requirements

| Component | Minimum | Used in development |
|-----------|---------|-------------------|
| GPU | Any CUDA | RTX 5060 8GB, CUDA 12.8 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB (dataset) | 484 BraTS cases |
| Python | 3.10+ | 3.11 |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/vicho141981/nexus-medical.git
cd nexus-medical

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install PySide6 vtk nibabel scipy reportlab

# 3. Download BraTS Task01_BrainTumour dataset
# Place at: data/Task01_BrainTumour/
# Dataset: http://medicaldecathlon.com/

# 4. Run the UI
python nexus_ui.py
python nexus_ui.py --base "data/Task01_BrainTumour"
```

**Note:** Use PySide6, not PyQt6 — PyQt6 causes DLL load failures on Windows.

---

## Usage

### Desktop UI
```bash
python nexus_ui.py --base "F:/NEXUS/archivos nexus medico/Task01_BrainTumour"
```
1. Select a BraTS case from the dropdown
2. Choose MRI modality (default: T1gd) and tumor label (default: Enhancing)
3. Set number of epochs (8000 for preview, 20000 for publication)
4. Click **ANALIZAR**
5. View 3D tumor + clinical result
6. Click **Guardar Reporte PDF** to export

### Batch pipeline (20 cases)
```bash
python nexus_brats_pipeline_v4.py --base "data/Task01_BrainTumour" --n_casos 20
```

### Scientific validation (vs Zhang et al. 2024)
```bash
python nexus_validacion_v7.py
```
Runs 8 synthetic cases (S1-S8) with square IC, 20000 epochs each.
Results saved to `runs_medico/v7_FINAL_*.json`.

### 3D Visualization
```bash
python nexus_viz_v4.py --base "data/Task01_BrainTumour" --caso BRATS_001 --D 0.00147 --rho 0.09226 --riesgo BAJO
```

---

## Validation results

Validated against **Zhang et al. (2024)** — *Personalized Predictions of Glioblastoma Infiltration*, Medical Image Analysis, DOI: 10.1016/j.media.2024.103423 (arXiv:2311.16536).

### Synthetic cases (S1-S8, IC cuadrada, 20k epochs)

| Case | D/ρ GT | err D/ρ% | Level |
|------|--------|----------|-------|
| S2 | 0.1875 | 53.5% | — |
| S4 | 0.1000 | 28.0% | MEDIA |
| S8 | 0.0714 | 5.8% | ALTA |

**Key finding:** D/ρ is robustly identifiable when D/ρ ∈ [0.07, 0.19] with single-modality MRI (error ~29% mean for identifiable cases). Individual D and ρ separation requires two MRI modalities (T1gd + FLAIR), following Zhang et al. 2024.

### Real BraTS cases (20 cases, pipeline v4)
| Parameter | Mean | Range |
|-----------|------|-------|
| D | 0.00276 | [0.00042, 0.00705] |
| ρ | 0.09854 | [0.00182, 0.45318] |
| D/ρ | 0.124 | [0.016, 0.386] |

---

## Scientific limitations

- **2D+t model:** Inference runs on the axial slice with maximum tumor area. Full 3D is future work.
- **Single modality:** Uses T1gd only. Two-modality (T1gd + FLAIR) would allow individual D and ρ identification.
- **Synthetic snapshots:** umid and u1 are simulated from u0. Real longitudinal MRI data would improve clinical validity.
- **Normalized domain:** D and ρ live in [0,1]×[0,1] normalized space. Conversion to physical units (mm²/day) requires voxel spacing calibration.

---

## Citation

If you use NEXUS Medical in your research, please cite:

```bibtex
@software{nexus_medical_2026,
  title   = {NEXUS Medical: Physics-Informed Neural Networks for Brain Tumor Analysis},
  year    = {2026},
  url     = {https://github.com/vicho141981/nexus-medical},
  note    = {Fisher-KPP PINN framework for tumor invasiveness estimation from BraTS MRI data}
}
```

Reference paper:
```bibtex
@article{zhang2024personalized,
  title   = {Personalized Predictions of Glioblastoma Infiltration from Sparse Spatiotemporal Measurements},
  author  = {Zhang, C. and others},
  journal = {Medical Image Analysis},
  year    = {2024},
  doi     = {10.1016/j.media.2024.103423}
}
```

---

## License

Copyright 2026 NEXUS Medical Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
