# NitroSpike: Nitrosative Susceptibility Mapping of Coronavirus Spike Proteins

## Overview
NitroSpike is a computational pipeline for analyzing coronavirus spike proteins and identifying nitrosative stress–susceptible residues.  
It integrates sequence analysis, structural features, and a biologically informed Nitrosative Susceptibility Index (NSI) to rank candidate sites.

The repository also reproduces all figures (Figure 2–5) from the study.

---

## Workflow
### 1. Data Preparation
- `01_check_and_combine_raw_fastas`  
  → FASTA quality check and inventory  
- `02_combine_fastafiles`  
  → combine all sequences into a single FASTA  

---
---

### 2. Paper Figures
- **Figure 2** → sequence structure and phylogeny  
- **Figure 3** → mutation, entropy, and hotspot analysis  
- **Figure 4** → variant-specific NSI and ΔNSI  
- **Figure 5** → interface analysis and residue overlap  
---
### 3. ML Pipeline (Run in Order)

- `ML01_curate_combined_fastafile`  
  → sequence curation, MAFFT alignment, Wuhan reference mapping  
- `ML02_extract_nitro_targets`  
  → extract Cys and Tyr target residues  
- `ML03_extract_structural_features`  
  → generate structural and physicochemical feature matrix  
- `ML04_generate_nsi_ranking_and_tree_sensitivity`  
  → compute NSI and rank candidate sites  
- `ML05_plot_validation`  
  → cross-variant validation and consistency analysis  

## Inputs
- Raw FASTA files:
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
