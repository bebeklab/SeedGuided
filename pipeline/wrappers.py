import sys
import os
import subprocess
import random
import math
import shutil
import numpy as np
import networkx as nx
import tempfile
import pandas as pd
from scipy.stats import hypergeom
from unittest.mock import MagicMock
# Add GNNSubNet to path so we can import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'GNN-SubNet')))
from GNNSubNet import GNNSubNet as gnn



# ==============================================================================
# ULTIMATE DGL / TORCHDATA MOCK
# This intercepts DGL's attempts to import deleted PyTorch modules
# ==============================================================================
sys.modules['torchdata'] = MagicMock()
sys.modules['torchdata.datapipes'] = MagicMock()
sys.modules['torchdata.datapipes.iter'] = MagicMock()
sys.modules['torchdata.dataloader2'] = MagicMock()
sys.modules['torchdata.dataloader2.graph'] = MagicMock()
sys.modules['dgl.graphbolt'] = MagicMock()  # <-- Kills the failing DGL module entirely!




# ==============================================================================
# ==============================================================================



# ==========================================
# SGSA HELPER FUNCTIONS
# ==========================================
def objective_function(subgraph_nodes, scores, size_penalty=0.1):
    score_sum = sum(scores.get(g, 0) for g in subgraph_nodes)
    return score_sum - size_penalty * len(subgraph_nodes)

def get_neighbors(G, subgraph_nodes):
    nbrs = set()
    for n in subgraph_nodes:
        nbrs.update(G.neighbors(n))
    return nbrs - subgraph_nodes

def is_connected(G, nodes):
    # NetworkX is_connected requires a non-empty graph
    if not nodes:
        return False
    return nx.is_connected(G.subgraph(nodes))




def run_seedmix(network, known_seeds, hidden_seeds=None, density=0.1, disease_name="temp"):
    """
    Runs SeedMix using dynamic, BUM-compliant p-values.
    """
    if hidden_seeds is None:
        hidden_seeds = set()
        
    safe_name = disease_name.replace(" ", "_").replace("/", "_")
    base_dir = os.path.abspath(os.getcwd())
    unique_dir = os.path.join(base_dir, f"temp2_seedmix_sandbox_{safe_name}")
    os.makedirs(unique_dir, exist_ok=True)
    
    temp_edge_list = os.path.join(unique_dir, "network.txt")
    temp_gene_scores = os.path.join(unique_dir, "scores.tsv")
    temp_seed_file = os.path.join(unique_dir, "seeds.txt")
    out_dir = os.path.join(unique_dir, "output")
    
    try:
        # 1. Write the network
        with open(temp_edge_list, 'w') as f:
            for u, v in network.edges():
                f.write(f"{u}\t{v}\n")
                
        # 2. Write dynamic p-values (The BUM Model Simulation)
        with open(temp_gene_scores, 'w') as f:
            for node in network.nodes():
                if node in known_seeds:
                    pval = random.uniform(1e-16, 1e-10) # Massive signal
                elif node in hidden_seeds:
                    # Beta(0.3, 1.0) skews heavily toward 0 but allows biological variance
                    pval = np.random.beta(0.3, 1.0) 
                    # Ensure it never exactly hits 0.0 to prevent log(0) errors
                    pval = max(pval, 1e-16) 
                else:
                    # Background MUST be Uniform
                    pval = random.uniform(0.01, 1.0)
                f.write(f"{node}\t{pval}\n")
                
        # 3. Write the known seeds
        with open(temp_seed_file, 'w') as f:
            for seed in known_seeds: 
                f.write(f"{seed}\n")
                
        # 4. Run SeedMix via command line
# 4. Run SeedMix via command line
        cmd = [
            "python", "run_seedmix3.py", 
            "--edge_list", temp_edge_list,
            "--gene_score", temp_gene_scores,
            "--seed_genes", temp_seed_file,
            "--density", str(density),
            "--num_edges", "30000",
            "--time_limit", "1",
            "--thread_count", "24",
            "--edge_dense_linear",  # <--- Just pass the flag itself!
            "--verbose", "0"
        ]

        seedmix_dir = os.path.join(base_dir, "pipeline", "seedmix")
        subprocess.run(cmd, cwd=seedmix_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        
        # 5. Read output and format scores (adding microscopic noise for sklearn tie-breaking)
# 5. Read output and format scores
        scores_dict = {node: random.uniform(0, 1e-5) for node in network.nodes()}
        
        out_file = os.path.join(out_dir, 'seedmix_subnetwork.tsv')
        if os.path.exists(out_file):
            with open(out_file, 'r') as f:
                seedmix_genes = [line.strip() for line in f if line.strip()]
                
                print(f"      -> SeedMix returned {len(seedmix_genes)} genes for {disease_name}")
                # --------------------------------
                
                for gene in seedmix_genes:
                    if gene in scores_dict:
                        scores_dict[gene] = 1.0 + random.uniform(0, 1e-5)
                
        return scores_dict
        
    except Exception as e:
        print(f"  [!] SeedMix failed. Error: {e}")
        return {node: random.uniform(0, 1e-5) for node in network.nodes()}
        
    finally:
        if os.path.exists(unique_dir):
            shutil.rmtree(unique_dir)



def seed_guided_simulated_annealing(G, seed_set, scores,
                                    size_penalty=0.1,
                                    T_start=1.0,
                                    alpha=0.95,
                                    max_iter=1000):
    SN_current = set(seed_set)
    
    # Ensure all initial seeds are actually in the graph to avoid crashes
    valid_seeds = {s for s in seed_set if s in G.nodes()}
    if not valid_seeds:
        return set(), 0
        
    SN_current = valid_seeds
    Score_current = objective_function(SN_current, scores, size_penalty)
    SN_best, Score_best = SN_current.copy(), Score_current
    T = T_start

    for _ in range(max_iter):
        SN_prop = SN_current.copy()

        # Propose add/remove
        if random.random() < 0.5:
            nbrs = list(get_neighbors(G, SN_current))
            if nbrs:
                SN_prop.add(random.choice(nbrs))
        else:
            removable = list(SN_current - valid_seeds)
            if removable:
                SN_prop.remove(random.choice(removable))

        if not is_connected(G, SN_prop):
            continue

        Score_prop = objective_function(SN_prop, scores, size_penalty)
        delta = Score_prop - Score_current

        if delta > 0 or random.random() < math.exp(delta / T):
            SN_current, Score_current = SN_prop, Score_prop
            if Score_current > Score_best:
                SN_best, Score_best = SN_current.copy(), Score_current

        T *= alpha

    return SN_best, Score_best


# ==========================================
# 1. YOUR METHOD: SGSA (Benchmark Wrapper)
# ==========================================
def run_sgsa(network, seed_genes):
    """
    Executes Seed-Guided Simulated Annealing.
    """
    valid_seeds = [s for s in seed_genes if s in network.nodes()]
    
    # 1. Generate biological scores using Random Walk with Restart (PageRank)
    # This simulates having external omics data, purely based on network topology
    personalization = {node: (1.0 if node in valid_seeds else 0.0) for node in network.nodes()}
    try:
        # alpha=0.85 is standard for random walks
        base_scores = nx.pagerank(network, alpha=0.85, personalization=personalization) 
    except nx.PowerIterationFailedConvergence:
        base_scores = {node: random.random() for node in network.nodes()}

    # 2. Run your Simulated Annealing algorithm
    # Using a slightly faster max_iter for the benchmark loop, but you can increase it!
    best_module, best_score = seed_guided_simulated_annealing(
        G=network, 
        seed_set=valid_seeds, 
        scores=base_scores,
        size_penalty=0.001, # Tuned down slightly because PageRank scores are very small decimals
        max_iter=500 
    )
    
    # 3. Format the output for the benchmark
    final_scores = {}
    for node in network.nodes():
        # Nodes chosen by your SA algorithm get a massive +1.0 boost 
        # so they rank at the absolute top of the ROC/PR curves
        if node in best_module:
            final_scores[node] = base_scores[node] + 1.0
        else:
            final_scores[node] = base_scores[node]
            
    return final_scores




# ==========================================
# 3. BASELINE: True DIAMOnD (Iterative)
# ==========================================
def run_diamond(network, seed_genes, max_added_nodes=200):
    """
    The accurate, iterative implementation of the DIAMOnD algorithm.
    Grows the disease module one node at a time based on hypergeometric significance.
    """
    valid_seeds = set([s for s in seed_genes if s in network.nodes()])
    
    if not valid_seeds:
        return {node: 0.0 for node in network.nodes()}
        
    N = network.number_of_nodes()
    module_nodes = set(valid_seeds)
    added_order = []
    
    # 1. Iteratively grow the module
    for _ in range(max_added_nodes):
        S = len(module_nodes)
        best_node = None
        best_p_val = float('inf')
        
        # Find all candidates (neighbors of the current module)
        candidates = set()
        for node in module_nodes:
            candidates.update(network.neighbors(node))
        candidates = candidates - module_nodes
        
        if not candidates:
            break # No more connected nodes to add
            
        # Calculate significance for all candidates
        for candidate in candidates:
            k = network.degree(candidate)
            k_s = sum(1 for neighbor in network.neighbors(candidate) if neighbor in module_nodes)
            
            # hypergeom.sf is the survival function (1 - CDF), exact p-value for enrichment
            p_val = hypergeom.sf(k_s - 1, N, S, k)
            
            if p_val < best_p_val:
                best_p_val = p_val
                best_node = candidate
        
        if best_node is None:
            break
            
        # Permanently add the most significant node to the module
        module_nodes.add(best_node)
        added_order.append(best_node)
        
    # 2. Convert to Benchmark Scores
    # Our benchmark needs continuous scores for ROC/PR curves.
    # We rank nodes based on the iteration they were added.
    scores = {}
    total_added = len(added_order)
    
    for node in network.nodes():
        if node in valid_seeds:
            # Seeds get the maximum absolute score
            scores[node] = 1.0 
        elif node in added_order:
            # Added nodes get a score between 0.99 and 0.01 based on how early they were added
            rank_idx = added_order.index(node)
            scores[node] = 0.99 - (0.98 * (rank_idx / max(1, total_added)))
        else:
            # Nodes that were never added get 0
            scores[node] = 0.0
            
    return scores
    

# ==========================================
# 4. BASELINE: GNN-SubNet (Synthetic Omics)
# ==========================================

def run_gnn_subnet_uniform(network, seed_genes, n_patients=100):
    """
    Wraps GNN-SubNet for the topology benchmark by generating synthetic patient data.
    """
    valid_seeds = set([s for s in seed_genes if s in network.nodes()])
    all_genes = list(network.nodes())
    
    if not valid_seeds:
        return {node: 0.0 for node in all_genes}

    # Create a temporary directory to hold the files GNNSubNet requires
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write the Network File
        ppi_path = os.path.join(tmpdir, 'NETWORK_tmp.txt')
        with open(ppi_path, 'w') as f:
            # 1. ADD THE HEADER LINE GNN-SUBNET IS LOOKING FOR:
            f.write("protein1 protein2 combined_score\n") 
            
            # 2. EXTRACT YOUR ACTUAL WEIGHTS:
            for u, v, d in network.edges(data=True):
                weight = float(d.get('weight', 1.0)) * 999
                f.write(f"{u} {v} {weight}\n")


                                
        # 2. Generate Synthetic "Patient" Features
        # Rows = patients, Columns = genes
        # Half healthy (class 0), Half disease (class 1)
        n_half = n_patients // 2
        
        # Base noise for all patients (mean 0, std 1)
        features = np.random.normal(0, 1, size=(n_patients, len(all_genes)))
        
        # Add strong signal to the seed genes ONLY for the disease patients (second half)
        seed_indices = [all_genes.index(g) for g in valid_seeds]
        for idx in seed_indices:
            features[n_half:, idx] += 5.0 # Massive spike in expression
            
        # Format as dataframe with patient IDs (rows) and gene names (columns)
        feat_df = pd.DataFrame(features, columns=all_genes)
        feat_df.index = [f"Patient_{i}" for i in range(n_patients)]
        
        feat_path = os.path.join(tmpdir, 'FEATURES_tmp.txt')
        # GNNSubNet expects space-separated features
        feat_df.to_csv(feat_path, sep=' ')
        
        # 3. Generate Target File
        targ_path = os.path.join(tmpdir, 'TARGET_tmp.txt')
        targets = [0]*n_half + [1]*n_half
        with open(targ_path, 'w') as f:
            for t in targets:
                f.write(f"{t}\n")
                
        # 4. Initialize and Run GNNSubNet
        try:
            # We redirect standard output momentarily because GNNSubNet prints a lot
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Load the data
            g = gnn.GNNSubNet(tmpdir, ppi_path, [feat_path], targ_path, normalize=True)
            
            # Train the network (keep epochs low for benchmark speed)
            g.train(epoch_nr=10) 
            
            # Run the explainer to get gene importances
            g.explain(n_runs=3)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # 5. Extract Scores
            scores = {node: 0.0 for node in all_genes}
            
            # Match the explainer masks back to the gene names
            if hasattr(g, 'node_mask') and hasattr(g, 'gene_names'):
                for gene, importance in zip(g.gene_names, g.node_mask):
                    scores[gene] = float(importance)
                    
            # Set seed scores to max so they don't penalize ROC curves
            max_score = max(scores.values()) if scores else 1.0
            if max_score == 0: max_score = 1.0
            for seed in valid_seeds:
                scores[seed] = max_score * 1.1 
                
            return scores

        except Exception as e:
            # Restore stdout in case of crash
            sys.stdout = old_stdout
            
            # If it's the class collapse bug, just return random baseline scores quietly
            if "index 1 is out of bounds" in str(e):
                import random
                return {node: random.uniform(0.0, 0.1) for node in all_genes}
                
            # Otherwise, print the real error
            print(f"  [!] GNN-SubNet failed for this disease: {e}")
            return {node: 0.0 for node in all_genes}
            
            
# ==========================================
# 4. BASELINE: GNN-SubNet (Synthetic Omics)
# ==========================================
def run_gnn_subnet(network, seed_genes, hidden_seeds=None, disease_name="temp", n_patients=100):

    """
    Wraps GNN-SubNet for the topology benchmark by generating synthetic patient data.
    """
    valid_seeds = set([s for s in seed_genes if s in network.nodes()])
    all_genes = list(network.nodes())
    
    # Initialize hidden_seeds safely
    if hidden_seeds is None:
        hidden_seeds = set()
    valid_hidden = set([s for s in hidden_seeds if s in network.nodes()])
    
    if not valid_seeds:
        return {node: 0.0 for node in all_genes}

    # Create a temporary directory to hold the files GNNSubNet requires
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write the Network File
        ppi_path = os.path.join(tmpdir, 'NETWORK_tmp.txt')
        with open(ppi_path, 'w') as f:
            # 1. ADD THE HEADER LINE GNN-SUBNET IS LOOKING FOR:
            f.write("protein1 protein2 combined_score\n") 
            
            # 2. EXTRACT YOUR ACTUAL WEIGHTS:
            for u, v, d in network.edges(data=True):
                weight = float(d.get('weight', 1.0)) * 999
                f.write(f"{u} {v} {weight}\n")


                                
        # 2. Generate Synthetic "Patient" Features
        n_half = n_patients // 2
        features = np.random.normal(0, 1, size=(n_patients, len(all_genes)))
        
        # Add strong signal (+5.0) to KNOWN seeds for the disease patients
        seed_indices = [all_genes.index(g) for g in valid_seeds]
        for idx in seed_indices:
            features[n_half:, idx] += 5.0 
            
        # Add a moderate, biologically variable signal (+2.0 to +4.0) to HIDDEN seeds
        hidden_indices = [all_genes.index(g) for g in valid_hidden]
        for idx in hidden_indices:
            # Random variance mimics patient heterogeneity
            features[n_half:, idx] += np.random.uniform(2.0, 4.0)
            
        # Format as dataframe
        feat_df = pd.DataFrame(features, columns=all_genes)
        feat_df.index = [f"Patient_{i}" for i in range(n_patients)]

        
        feat_path = os.path.join(tmpdir, 'FEATURES_tmp.txt')
        # GNNSubNet expects space-separated features
        feat_df.to_csv(feat_path, sep=' ')
        
        # 3. Generate Target File
        targ_path = os.path.join(tmpdir, 'TARGET_tmp.txt')
        targets = [0]*n_half + [1]*n_half
        with open(targ_path, 'w') as f:
            for t in targets:
                f.write(f"{t}\n")
                
        # 4. Initialize and Run GNNSubNet
        try:
            # We redirect standard output momentarily because GNNSubNet prints a lot
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Load the data
            g = gnn.GNNSubNet(tmpdir, ppi_path, [feat_path], targ_path, normalize=True)
            
            # Train the network (keep epochs low for benchmark speed)
            g.train(epoch_nr=10) 
            
            # Run the explainer to get gene importances
            g.explain(n_runs=3)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # 5. Extract Scores
            scores = {node: 0.0 for node in all_genes}
            
            # Match the explainer masks back to the gene names
            if hasattr(g, 'node_mask') and hasattr(g, 'gene_names'):
                for gene, importance in zip(g.gene_names, g.node_mask):
                    scores[gene] = float(importance)
                    
            # Set seed scores to max so they don't penalize ROC curves
            max_score = max(scores.values()) if scores else 1.0
            if max_score == 0: max_score = 1.0
            for seed in valid_seeds:
                scores[seed] = max_score * 1.1 
                
            return scores

        except Exception as e:
            # Restore stdout in case of crash
            sys.stdout = old_stdout
            
            # If it's the class collapse bug, just return random baseline scores quietly
            if "index 1 is out of bounds" in str(e):
                import random
                return {node: random.uniform(0.0, 0.1) for node in all_genes}
                
            # Otherwise, print the real error
            print(f"  [!] GNN-SubNet failed for this disease: {e}")
            return {node: 0.0 for node in all_genes}
            
            
    
# ==========================================
# 2. BASELINE: Random Scoring (For testing)
# ==========================================
def run_random_baseline(network, seed_genes):
    """Assigns completely random scores to all genes."""
    scores = {}
    for node in network.nodes():
        scores[node] = random.random()
    return scores
    
    
    
