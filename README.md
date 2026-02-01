# ExSEnt for explainable dementia detection

## 1. Introduction

This repository contains the analysis code supporting the methodology and results reported in the manuscript:

Sara Kamali, Fabiano Baroni, Pablo Varona,
"ExSEnt for explainable dementia detection: disentangling temporal and amplitude-driven complexity boosts EEG-based classification".

We used a publicly available resting-state, eyes-closed EEG dataset encoded in BIDS (Brain Imaging Data Structure):

A. Miltiadous, et al., "A dataset of scalp EEG recordings of Alzheimer’s disease, frontotemporal dementia and healthy subjects from routine EEG", Data 8 (2023) 95. https://doi.org/10.3390/data8060095.

Pipeline summary:
1. Subject-level preprocessing and independent component (IC) extraction.
2. Source localization and selection of brain-related ICs.
3. Group-level IC clustering and selection of four clusters for downstream analysis:
   LVA, RVA, LPFC, RPFC (left/right visual association; left/right prefrontal cortex).
4. Band-limited signal and power extraction from cluster IC time series.
5. Sliding-window feature extraction (complexity and spectral features).
6. Stability selection and penalized logistic regression to identify stable, explainable biomarkers for early dementia detection.

## 2. File descriptions and intended order of use

1. `Preprocessing_raw_EEG.m`  
   Preprocesses raw EEG, computes IC decomposition, performs source localization, and identifies brain-related ICs (good ICs) for each subject.

2. `STUDY_Genetator.m`  
   Creates a group-level EEGLAB STUDY, clusters ICs across subjects, and retains at most one IC per subject per cluster.

3. `bandpass_IC_signals.m`  
   Extracts band-passed IC signals and band-specific power using `cwt`/`icwt`. Calls `cwt_kaiser_filter.m` to perform filtering.

4. `cwt_kaiser_filter.m`  
   Helper function implementing the wavelet-based filtering used by `bandpass_IC_signals.m`.

5. `EEG_feature_extraction.m`  
   Computes complexity and spectral features over 23 consecutive windows with 50% overlap for each subject/cluster/band. Feature functions include:
   1. `Higuchi_FD.m`
   2. `KatzFD.m`
   3. `hurst_exponent.m`
   4. `sample_entropy.m`
   5. `ExSEnt.m` (calls `extract_DA.m` and `sample_entropy.m`)
   6. `extract_DA.m`

6. `prep_feature.m`  
   Prepares feature tensors for downstream Python analysis. Calls `export_dementia_tensors_pure_feats.m`.

7. `export_dementia_tensors_pure_feats.m`  
   Exports feature tensors in a format consumed by the Python stability selection pipeline.

8. `Stability_Selection_main.py`  
   Runs stability selection over extracted features to identify stable and interpretable biomarkers, and fits penalized logistic regression models for classification and reporting.

## 3. Requirements (high level)

MATLAB:
1. EEGLAB (for preprocessing, IC handling, and STUDY clustering).
2. Wavelet functions (`cwt`, `icwt`) as used in the band-pass workflow.

Python:
1. Python 3.x.
2. Standard scientific stack (NumPy, SciPy, pandas, scikit-learn) for stability selection and classification.

## 4. Recommended repository structure

1. `data/` for BIDS dataset (not included in this repository).
2. `matlab/` for MATLAB scripts.
3. `python/` for Python scripts.
4. `results/` for exported tensors, tables, and figures.

## 5. Citation

If you use this code, cite:
1. The dataset paper: doi:10.3390/data8060095.
2. The manuscript listed in Section 1 (when publicly available).

## 6. Contact

Sara Kamali (sara.kamali@gmail.com)
