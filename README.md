# SL2301_HeatAdaptation

Repository for the Heat Acclimation office-intervention study investigating physiological and perceptual adaptation to different thermal exposure patterns in office environments.

---

# Project summary

This repository contains processed datasets, preprocessing scripts, analysis pipelines, figures, and documentation associated with the Heat Acclimation study conducted at the TUM SenseLab.

The project investigated whether repeated exposure to free-running warm office environments alters thermophysiological regulation compared with continuous air-conditioned exposure.

Participants completed:

- Pre-intervention heat-stress testing (HS1)
- Multi-day office intervention exposure
- Post-intervention heat-stress testing (HS2)

Two intervention arms were studied:

- **FR** — free-running office environment
- **CC / AC** — continuously cooled office environment

Primary outcomes included:

- Core body temperature (CBT)
- Skin temperature
- Heart-rate variability (HRV)
- Blood pressure
- Thermal perception questionnaires
- Environmental exposure metrics

---

# Repository structure

```text
SL2301_HeatAdaptation/
│
├── code/
│   ├── 00_preprocessing/
│   └── 03_analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── outputs/
│   ├── climate/
│   └── model_outputs/
│
├── docs/
│
├── DATA_DICTIONARY.md
├── LICENSE
└── README.md
```

---

# Data availability

This repository contains processed datasets intended for reproducible analyses.

Raw physiological recordings are not distributed publicly because of:

- participant privacy
- large file sizes
- proprietary export formats
- device-specific software dependencies

---

# Statistical approach

Primary analyses generally used linear mixed-effects models.

Typical model structure:

```text
outcome ~ arm * scenario + sex_c + (1 | part_id)
```

Where:

- `arm` = intervention arm
- `scenario` = HS1 vs HS2
- `sex_c` = coded sex covariate
- `part_id` = participant random intercept

---

# Reproducibility notes

All scripts use repository-relative paths via:

```python
Path(__file__).resolve().parents[2]
```

Recommended environment:

- Python ≥ 3.10
- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- neurokit2
- patsy
- tqdm

---

# Citation

If using this repository or associated datasets, please cite the corresponding manuscript(s).

---

# Contact

Bilge Kobas  
Technical University of Munich (TUM)  
Chair of Building Technology and Climate Responsive Design  
TUM SenseLab
