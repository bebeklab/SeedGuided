# ==============================================================================
#  DGL / TORCHDATA MOCK
# ==============================================================================
import sys
from unittest.mock import MagicMock
import shutil

# MagicMock automatically fakes ANY requested class/function (like 'Mapper')
sys.modules['torchdata'] = MagicMock()
sys.modules['torchdata.datapipes'] = MagicMock()
sys.modules['torchdata.datapipes.iter'] = MagicMock()
sys.modules['torchdata.dataloader2'] = MagicMock()
sys.modules['torchdata.dataloader2.graph'] = MagicMock()

# This kills the failing DGL submodule entirely so it stops looking for torchdata
sys.modules['dgl.graphbolt'] = MagicMock() 
# ==============================================================================
import types



# ==============================================================================
import os
import random
import networkx as nx
import pandas as pd
import numpy as np
import concurrent.futures
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from pipeline.wrappers import run_sgsa, run_sgsa_multirun, run_random_baseline, run_diamond, run_gnn_subnet, run_seedmix


def load_data_OMNI_STRING():
    """Loads the Harmonizome network and disease gene lists directly from .gz files."""
    print("Loading Harmonizome Network (OMNI-STRING PPI)...")
    net_path = "~/benchmark/disease_benchmark/pipeline/seedmix/data/PPI_2021_network.txt"
# usecols=[0, 1] forces pandas to just take the first two columns regardless of what they are named
    net_df = pd.read_csv(net_path, sep='\t', skiprows=[0], header=None, usecols=[0, 1])
    net_df.columns = ["A", "B"]    
    G = nx.from_pandas_edgelist(net_df, source="A", target="B")
    
    print("Loading Harmonizome Diseases (Curated)...")
    dis_path = "data/DISEASES Curated Gene-Disease Assocation Evidence Scores/gene_attribute_edges.txt.gz"
    
    dis_df = pd.read_csv(dis_path, sep='\t', skiprows=[1], usecols=['source', 'target'])
    dis_df.columns = ["Gene", "Disease"]
    
    diseases = dis_df.groupby("Disease")["Gene"].apply(set).to_dict()
    
    # Filter diseases to only keep those where the genes actually exist in our network
    network_nodes = set(G.nodes())
    clean_diseases = {}
    for disease, genes in diseases.items():
        valid_genes = set([g for g in genes if g in network_nodes])
        if 20 <= len(valid_genes) <= 300:
        #if len(valid_genes) >= 20: # Minimum module size 
            clean_diseases[disease] = valid_genes
            
    print(f"Loaded network with {G.number_of_nodes()} genes and {G.number_of_edges()} edges.")
    print(f"Found {len(clean_diseases)} valid diseases to benchmark.")
    
    return G, clean_diseases


def load_data_INTACT():
    """Loads the Harmonizome network and disease gene lists directly from .gz files."""
    print("Loading Harmonizome Network (IntAct PPI)...")
    net_path = "data/IntAct Biomolecular Interactions/gene_attribute_edges.txt.gz"
    
    net_df = pd.read_csv(net_path, sep='\t', skiprows=[1], usecols=['source', 'target'])
    net_df.columns = ["Gene", "Target"] 
    G = nx.from_pandas_edgelist(net_df, source="Gene", target="Target")
    
    print("Loading Harmonizome Diseases (Curated)...")
    dis_path = "data/DISEASES Curated Gene-Disease Assocation Evidence Scores/gene_attribute_edges.txt.gz"
    
    dis_df = pd.read_csv(dis_path, sep='\t', skiprows=[1], usecols=['source', 'target'])
    dis_df.columns = ["Gene", "Disease"]
    
    diseases = dis_df.groupby("Disease")["Gene"].apply(set).to_dict()
    
    # Filter diseases to only keep those where the genes actually exist in our network
    network_nodes = set(G.nodes())
    clean_diseases = {}
    for disease, genes in diseases.items():
        valid_genes = set([g for g in genes if g in network_nodes])
        if 20 <= len(valid_genes) <= 300:
        #if len(valid_genes) >= 20: # Minimum module size 
            clean_diseases[disease] = valid_genes
            
    print(f"Loaded network with {G.number_of_nodes()} genes and {G.number_of_edges()} edges.")
    print(f"Found {len(clean_diseases)} valid diseases to benchmark.")
    
    return G, clean_diseases

def load_data_DIP():
    """Loads the Harmonizome network and disease gene lists directly from .gz files."""
    print("Loading Harmonizome Network (DIP PPI)...")
    net_path = "data/DIP Protein-Protein Interactions/gene_attribute_edges.txt.gz"
    
    net_df = pd.read_csv(net_path, sep='\t', skiprows=[1], usecols=['source', 'target'])
    net_df.columns = ["Gene", "Target"] 
    G = nx.from_pandas_edgelist(net_df, source="Gene", target="Target")
    
    print("Loading Harmonizome Diseases (Curated)...")
    dis_path = "data/DISEASES Curated Gene-Disease Assocation Evidence Scores/gene_attribute_edges.txt.gz"
    
    dis_df = pd.read_csv(dis_path, sep='\t', skiprows=[1], usecols=['source', 'target'])
    dis_df.columns = ["Gene", "Disease"]
    
    diseases = dis_df.groupby("Disease")["Gene"].apply(set).to_dict()
    
    # Filter diseases to only keep those where the genes actually exist in our network
    network_nodes = set(G.nodes())
    clean_diseases = {}
    for disease, genes in diseases.items():
        valid_genes = set([g for g in genes if g in network_nodes])
        if 20 <= len(valid_genes) <= 300:
            clean_diseases[disease] = valid_genes
            
    print(f"Loaded network with {G.number_of_nodes()} genes and {G.number_of_edges()} edges.")
    print(f"Found {len(clean_diseases)} valid diseases to benchmark.")
    
    return G, clean_diseases



def evaluate_single_disease(args):
    """Worker function to process one disease on a single CPU core."""
    # Unpack the new index and total variables
    task_idx, total_tasks, disease_name, all_disease_genes, network, algorithms = args
    
    print(f"[{task_idx}/{total_tasks}] [{disease_name}] Starting benchmark...")
        
    # 1. Train/Test Split: Hide 50% of the disease genes to test the algorithm
    all_disease_genes = list(all_disease_genes)
    random.shuffle(all_disease_genes)
    split_idx = len(all_disease_genes) // 2
    
    known_seeds = all_disease_genes[:split_idx]
    hidden_seeds = set(all_disease_genes[split_idx:])
    
    # The nodes we will actually evaluate on (excluding the known seeds)
    eval_nodes = [node for node in network.nodes() if node not in known_seeds]
    
    # Ground truth labels for evaluation
    y_true = [1 if node in hidden_seeds else 0 for node in eval_nodes]
    
    results = []
    
    # 2. Run each algorithm
    for algo_name, algo_func in algorithms.items():
        try:
            if algo_name == "SeedMix":
                # SeedMix uses hidden_seeds only for realistic p-value simulation (not as labels)
                scores_dict = algo_func(network, known_seeds, hidden_seeds=hidden_seeds, disease_name=disease_name)
            else:
                # All other methods — including GNN-SubNet — see only known seeds (no leakage)
                scores_dict = algo_func(network, known_seeds)
                
                
            # Extract scores for the evaluation nodes
            y_scores = [scores_dict.get(node, 0.0) for node in eval_nodes]
            
            # 3. Calculate Metrics
            roc_auc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            
            # For F1, we predict '1' for the top-K highest scoring genes (K = number of hidden seeds)
            top_k_indices = np.argsort(y_scores)[-len(hidden_seeds):]
            y_pred = np.zeros_like(y_true)
            y_pred[top_k_indices] = 1
            f1_val = f1_score(y_true, y_pred)
            
            results.append({
                "Disease": disease_name,
                "Algorithm": algo_name,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "F1_Score": f1_val
            })
            # Add the numbers to the success print
            print(f"[{task_idx}/{total_tasks}] [{disease_name}] {algo_name} finished. (ROC: {roc_auc:.3f})")
            
        except Exception as e:
            # Add the numbers to the error print
            print(f"[{task_idx}/{total_tasks}] [{disease_name}] Error in {algo_name}: {e}")
            
         
            
    return results

def main():
#    network, diseases = load_data_OMNI_STRING()
    network, diseases = load_data_INTACT()
    
    # Uncomment if you want to test on just a few diseases first
    # diseases = dict(list(diseases.items())[:3])
    
    # Define the algorithms to run (SeedMix is now UNCOMMENTED)
    algorithms = {
        "Random Baseline": run_random_baseline,
        "DIAMOnD": run_diamond,
        "SGSA (Yours)": run_sgsa_multirun,  # multi-run averaged for variance stability
        "GNN-SubNet": run_gnn_subnet,
        "SeedMix": run_seedmix
    }

    print(f"\nStarting parallel benchmark for {len(diseases)} diseases using {len(algorithms)} algorithms...")
    
    all_results = []
    total_tasks = len(diseases)
    tasks = [(i + 1, total_tasks, disease, seeds, network, algorithms) for i, (disease, seeds) in enumerate(diseases.items())]
    
    # Spin up the parallel processes. 
    max_cores = min(15, len(diseases)) 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        for disease_results in executor.map(evaluate_single_disease, tasks):
            all_results.extend(disease_results)
            
    # Save the final compiled dataframe
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/benchmark_metrics.csv", index=False)
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETE!")
    print("="*50)
    summary = results_df.groupby("Algorithm")[["ROC_AUC", "PR_AUC", "F1_Score"]]
    means = summary.mean().round(3)
    stds = summary.std().round(3)
    counts = summary.count()

    print("\nMean Performance:")
    print(means.to_string())
    print("\nStd Dev across diseases:")
    print(stds.to_string())
    print("\n95% Confidence Intervals (mean ± 1.96*std/sqrt(n)):")
    import math
    for algo in means.index:
        n = counts.loc[algo, "ROC_AUC"]
        for metric in ["ROC_AUC", "PR_AUC", "F1_Score"]:
            m = means.loc[algo, metric]
            s = stds.loc[algo, metric]
            ci = 1.96 * s / math.sqrt(n)
            print(f"  {algo:<22} {metric}: {m:.3f} ± {ci:.3f}")

    print("\nPer-Disease Variance (ROC_AUC):")
    per_disease = results_df.pivot_table(index="Disease", columns="Algorithm", values="ROC_AUC")
    print(per_disease.var().round(4).to_string())

    print("\nFull Per-Disease Results:")
    print(results_df.pivot_table(index="Disease", columns="Algorithm", values="ROC_AUC").round(3).to_string())

if __name__ == "__main__":
    main()
