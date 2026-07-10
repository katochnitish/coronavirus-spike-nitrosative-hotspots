# Coronavirus-spike-nitrosative-hotspots
Mapping nitrosative susceptibility hotspots in coronavirus spike proteins using AI and structural bioinformatics.
GitHub Release: v1.0.0


# NitroSpike: Nitrosative Susceptibility Mapping Pipeline

## Overview

NitroSpike is a computational pipeline for identifying nitrosative stress–susceptible residues in coronavirus spike proteins. It combines sequence alignment, structural feature extraction, and a biologically informed scoring model (NSI) to rank candidate sites.

---

## Pipeline

Run scripts in order:

**ML01** – Sequence curation and alignment

* Cleans FASTA, removes low-quality and duplicate sequences
* Aligns sequences (MAFFT) and converts to Wuhan reference coordinates

**ML02** – Target extraction

* Identifies Cysteine and Tyrosine residues
* Builds residue-level matrix and basic plots

**ML03** – Feature engineering

* Maps residues to structures (PDB)
* Computes structural, chemical, and evolutionary features

**ML04** – NSI scoring

* Computes Nitrosative Susceptibility Index using 4 components:
  accessibility, chemistry, evolution, functional context
* Produces ranked candidate sites and feature importance

**ML05 / ML09** – Validation

* Cross-variant testing and GroupKFold validation
* Evaluates model consistency and ranking stability

---

## Inputs

* FASTA: `combined_all_raw_sequences_curated.fasta`
* Generated feature matrix: `structural_ai_feature_matrix.csv`
* PDB structures (auto-downloaded or cached)

---

## Requirements

### External tools

* **MAFFT** (required for alignment)

### Python libraries

```
numpy pandas matplotlib seaborn scikit-learn biopython xgboost scipy
```

---

## Outputs

* Aligned FASTA files
* Residue-level feature matrix
* NSI-ranked candidate sites
* Validation reports and plots

---

## Notes

* NSI is a **proxy score**, not experimental ground truth
* ML is used for **ranking and sensitivity analysis**
* Validation is **cross-variant, not experimental**

---

## Usage

Run scripts sequentially from ML01 to ML09. Each step generates inputs for the next.
