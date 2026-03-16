# SeedGuided SA: Seed-Guided Simulated Annealing for Disease Subnetwork Discovery

This repository contains the source code, benchmarking pipeline, and dataset processing scripts for the manuscript: **"Seed-Driven Scalable Discovery of Dense Disease Modules for Enhanced Biomarker Identification"**.

## Overview
Identifying dense, disease-specific subnetworks within large protein-protein interaction (PPI) networks is crucial for unraveling complex disease mechanisms. While exact global optimization methods, such as Mixed-Integer Quadratic Programming (MIQP), provide rigorous mathematical guarantees, they suffer from a severe $O(N^2)$ combinatorial explosion on modern, dense human interactomes. 

**Seed-Guided Simulated Annealing (SGSA)** is a highly scalable stochastic metaheuristic that bypasses global matrix evaluations entirely. By initiating from a validated set of disease seeds and dynamically exploring local topological neighborhoods, SGSA balances signal enrichment against a strict size penalty to successfully escape local optima. 

In a comprehensive empirical benchmark across 102 curated diseases on the 12,000-node IntAct interactome, SGSA demonstrated superior biological signal recovery, significantly outperforming classic topological traversals (DIAMOnD), deep learning frameworks (GNN-SubNet), and exact optimization relaxations (SeedMix).

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
