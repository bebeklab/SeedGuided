#!/bin/bash
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:30:00     # 2:30 minutes
#SBATCH --mail-user=gxb43@case.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="netmix3_a_test"
#SBATCH -o %j.out 
#SBATCH -e %j.err

# Put commands for executing job below this line
# This example is loading the default Python module and then
# writing out the version of Python
module load Python


cd ~/seedmix
mkdir results_a1
python run_seedmix3.py --edge_list  data/PPI_2021_netmix.txt --gene_score data/gene_scores.tsv --seed_genes data/seed_genes.txt --density 0.1 --output results_a1 --verbose 1
