#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --job-name="gwas_seedmix_comparison"
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rxg796@case.edu

module load Python
export GRB_LICENSE_FILE=/home/rxg796/gurobi_wls.lic
cd ~/SeedGuided
pip install pandas
python3.11 run_gwas_comparison.py \
  --gwas "Rheumatoid Arthritis:GWAS Data/RheumatoidArthritis.h.tsv.gz" \
  --gwas "Type 2 Diabetes:GWAS Data/Type2Diabetes.h.tsv.gz" \
  --gwas "Bipolar Disorder:GWAS Data/pgc-bip2021-all.vcf.tsv.gz" \
  --gene-coords data/gene_coords_hg38.tsv \
  --window 10000 \
  --reference-genes "data/DISEASES Curated Gene-Disease Assocation Evidence Scores/gene_attribute_edges.txt.gz" \
  --reference-label "Rheumatoid Arthritis:rheumatoid arthritis" \
  --reference-label "Type 2 Diabetes:diabetes mellitus" \
  --gwas-catalog-file "GWAS Data/gwas-catalog-download-associations-v1.0-full.tsv" \
  --gwas-catalog-trait "Bipolar Disorder:=Bipolar disorder" \
  --network data/PCNet.txt \
  --num-edges-fraction 0.4 \
  --time-limit 8 \
  --seedmix-dir pipeline/seedmix \
  --outdir results_gwas_comparison_pcnet
