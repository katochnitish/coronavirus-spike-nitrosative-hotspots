# NitroSpike: Nitrosative Susceptibility Mapping of Coronavirus Spike Proteins

## Overview

NitroSpike is a computational framework for identifying and prioritizing nitrosative stress–susceptible residues in coronavirus spike proteins. The pipeline integrates sequence alignment, structural feature extraction, and a biologically grounded scoring model to compute a Nitrosative Susceptibility Index (NSI).

---

## Pipeline Structure
The workflow is organized into sequential scripts:

### 01 – FASTA Curation and Alignment (ML01)
* Cleans raw spike protein sequences
* Removes low-quality sequences (>1 percent unknown residues)
* Removes exact duplicates
* Performs multiple sequence alignment (MAFFT)
* Converts alignment into Wuhan-Hu-1 coordinate space

### 02 – Target Residue Extraction (ML02)
* Identifies biologically relevant residues:
  * Cysteine (S-nitrosylation)
  * Tyrosine (nitration)
* Extracts residue-level matrix across variants
* Generates conservation heatmaps and distribution plots

### 03 – Structural Feature Engineering (ML03)
* Maps residues onto protein structures (PDB)
* Computes physicochemical and structural features:
  * Solvent accessibility (SASA proxies)
  * Electrostatic environment
  * Local residue composition
  * Evolutionary metrics (entropy, conservation)
  * Spatial proximity (ACE2 interface, glycan shielding)
* Outputs machine learning–ready feature matrix

### 04 – NSI Scoring and Ranking (ML04)
* Computes hierarchical Nitrosative Susceptibility Index (NSI)
* Combines four biological components:
  * Accessibility
  * Chemistry
  * Evolution
  * Functional context
* Produces:
  * Variant-level rankings
  * Residue-level rankings
  * Feature ablation analysis
  * Tree-based sensitivity analysis
### 05 – Validation (ML05 & ML09)

* ML05:
  * Tests model generalization across variants
  * Uses rule-derived susceptibility labels (consistency check)
* ML09:
  * Performs GroupKFold cross-validation
  * Implements strict variant-level holdout (e.g., Delta, Omicron)
  * Evaluates prediction of NSI scores (internal validation)
---

## Input Data Requirements

### Primary Inputs

* Combined spike protein FASTA file
  `02_CuratedRawData/combined_all_raw_sequences_curated.fasta`

* Structural feature matrix (generated internally)
  `09_AI_NitroSpike/structural_ai_feature_matrix.csv`

### Structural Data

* Protein structures (PDB) are automatically downloaded or cached
* Used for residue-level feature extraction

---

## External Tools Required
### 1. MAFFT (Mandatory)
Used for multiple sequence alignment
Install:
* Windows: use `mafft.bat`
* Linux/WSL: ensure MAFFT is accessible via PATH
### 2. Python (>= 3.9)
### 3. Required Python Libraries
Install using:
```
pip install numpy pandas matplotlib seaborn scikit-learn biopython xgboost scipy
```

#### Core libraries used:
* numpy, pandas → data processing
* matplotlib, seaborn → plotting
* scikit-learn → ML pipeline, validation
* xgboost → tree-based models
* biopython → sequence and structure handling
* scipy → statistical analysis
---

## How to Run the Pipeline
Run scripts in order:
```
ML01_curate_combined_fastafile.py
ML02_extract_nitro_targets.py
ML03_extract_structural_features.py
ML04_generate_nsi_ranking_and_tree_sensitivity.py
ML05_external_variant_validation.py
ML09_plot_validation.py
```
Each script produces outputs used by the next stage.
---
## Outputs
Key outputs include:
* Curated aligned FASTA files
* Residue-level target matrices
* Structural feature dataset
* NSI-ranked candidate sites
* Feature importance and ablation reports
* Validation metrics and plots

## Reproducibility
* Fixed random seeds are used in ML models
* Group-based splitting prevents data leakage across variants
* All intermediate files are saved for traceability
