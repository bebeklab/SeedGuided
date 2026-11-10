# Seed-Driven Scalable Discovery of Dense Disease Modules for Enhanced Biomarker Identification

This repository contains the source code, benchmarking pipeline, and dataset processing scripts for the manuscript: **"Seed-Driven Scalable Discovery of Dense Disease Modules for Enhanced Biomarker Identification"**.

## Overview

Identifying disease-specific subnetworks within large protein-protein interaction (PPI) networks is a standard approach for understanding disease biology and discovering clinical biomarkers. Exact optimization methods such as Mixed-Integer Quadratic Programming (MIQP) can guarantee dense network modules, but on dense human interactomes global MIQP formulations generate millions of variables and quickly become computationally intractable.

**SeedMix** is a seed-guided, density-constrained subnetwork discovery method. Instead of evaluating the entire network adjacency matrix at once, SeedMix uses Personalized PageRank to reduce the search space to a biologically relevant neighborhood anchored by known disease seed genes. Within this reduced subgraph, a tractable Mixed-Integer Programming formulation balances topological cohesion against strict size constraints.

SeedMix was benchamrked across 102 curated diseases on the 12,000-node IntAct interactome, where it consistently outperformed established topological methods (DIAMOnD), graph neural networks (GNN-SubNet), and relaxed optimization approaches. SeedMix was also applied to patient-specific gene expression data in glioma, where it identified dysregulated pathways that accurately stratified clinical subgroups.

## Repository Structure
* `/pipeline/` - Contains the core `sgsa.py` algorithm alongside wrappers for the baseline methods (DIAMOnD, GNN-SubNet, SeedMix).
* `/data/` - Directory for input networks, synthetic gene scores, and curated seed sets. *(Note: Large interactome files are not hosted in this repo. See Data Availability below).*
* `/results/` - Output directory for benchmark metrics, subnetwork edge lists, and performance logs.
* `run_benchmark.py` - The primary execution script used to evaluate the 102 curated diseases.
* `requirements.txt` - Python dependencies required to replicate the environment.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/bebeklab/SeedGuided.git](https://github.com/bebeklab/SeedGuided.git)
cd SeedGuided
pip install -r requirements.txt




## License and Copyright
© 2026 Gurkan Bebek. All Rights Reserved. 
This code is provided for peer-review purposes only. No license is granted for reproduction, distribution, or modification of this code without explicit written permission.
