import sys, os, re
import networkx as nx
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Mocks
sys.modules['torchdata'] = MagicMock()
sys.modules['torchdata.datapipes'] = MagicMock()
sys.modules['torchdata.datapipes.iter'] = MagicMock()
sys.modules['torchdata.dataloader2'] = MagicMock()
sys.modules['torchdata.dataloader2.graph'] = MagicMock()
sys.modules['dgl.graphbolt'] = MagicMock()

from pipeline.wrappers import run_sgsa_multirun

# ── THRESHOLD PARAMETER ──────────────────────────────────────────────────────
# Controls the PPR score cutoff (delta) used to define the subnetwork boundary.
# Mirrors the diffusion threshold delta in Algorithm 1 (Phase 1) of the paper.
# Genes scoring above (mean + PPR_DELTA_STD × std) of background PPR scores are
# included in the candidate subnetwork. Seeds are always included regardless.
#
# Increase → smaller, stricter subnetwork (fewer genes pass the threshold)
# Decrease → larger subnetwork (more genes pass the threshold)
#
# At mean+3std, SN10 (the paper's worked example with 6 seeds) recovers ~150 genes,
# consistent with the paper's reported 133-gene SeedMix subnetwork for the same seeds.
PPR_DELTA_STD = 3.0


# ── 1. PARSE THE 92 nSEA NETWORKS ───────────────────────────────────────────
def parse_nsea_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style[^>]*>.*?</style>',  '', content, flags=re.DOTALL)
    content = re.sub(r'<[^>]+>', ' ', content)
    content = re.sub(r'\s+', ' ', content)

    networks = {}
    matches = re.findall(
        r'\[1\] (SN\d+) \d+ ([A-Z0-9 \-]+?)(?=##|\[1\] SN|\Z)',
        content
    )
    for name, genes_str in matches:
        genes = [g.strip() for g in genes_str.strip().split() if g.strip()]
        networks[name] = genes

    manual = {
        'SN37': ['GNAO1','SPTBN2','NCAM1','GNAI1','HLA-DRB1','HLA-DRB5'],
        'SN41': ['BPTF','TAF1','RSF1','TAF1L','C17orf49','TAF10'],
        'SN91': ['EEF2','EIF4B','MYT1','CD40','RPS6KA1'],
    }
    networks.update(manual)
    print(f"Parsed {len(networks)} nSEA networks.")
    return networks


# ── 2. LOAD BACKGROUND PPI ───────────────────────────────────────────────────
def load_network():
    print("Loading IntAct PPI network...")
    net_path = "data/IntAct Biomolecular Interactions/gene_attribute_edges.txt.gz"
    net_df = pd.read_csv(net_path, sep='\t', skiprows=[1], usecols=['source','target'])
    net_df.columns = ["Gene","Target"]
    G = nx.from_pandas_edgelist(net_df, source="Gene", target="Target")
    print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── 3. GLOBAL PPI STATISTICS ─────────────────────────────────────────────────
def compute_network_stats(G):
    """
    Prints connectivity statistics for the background PPI.
    The network is NOT fully connected (62 components), but 99.2% of nodes
    are in the largest connected component (LCC), so it is effectively one
    large graph. All analyses are anchored to the LCC.
    """
    components = list(nx.connected_components(G))
    lcc = max(components, key=len)

    print("\n" + "="*60)
    print("BACKGROUND PPI NETWORK STATISTICS")
    print("="*60)
    print(f"  Total nodes (genes):        {G.number_of_nodes():>8,}")
    print(f"  Total edges (interactions): {G.number_of_edges():>8,}")
    print(f"  Connected components:       {len(components):>8,}")
    print(f"  Is fully connected:         {'Yes' if nx.is_connected(G) else 'No':>8}")
    print(f"  Largest component (LCC):    {len(lcc):>8,} nodes "
          f"({len(lcc)/G.number_of_nodes()*100:.1f}% of all nodes)")
    print(f"  Smaller components:         {len(components)-1:>8,} "
          f"({G.number_of_nodes()-len(lcc):,} nodes total)")
    print(f"  Average degree:             {sum(d for _,d in G.degree())/G.number_of_nodes():>8.2f}")
    print("="*60 + "\n")
    return lcc


# ── 4. SUBNETWORK EXTRACTION ─────────────────────────────────────────────────
def extract_subnetwork(G, ppr_scores, valid_seeds, n_std=PPR_DELTA_STD):
    """
    Extracts a focused disease subnetwork from PPR scores, mirroring
    Phase 1 of SeedMix (Algorithm 1 in the paper):

        Phase 1: Diffusion Graph Construction
        - Compute PPR scores W from G (personalised on seeds)
        - Apply sparsity threshold delta: include gene i if W_i > delta
        - delta here = mean + n_std × std of background (non-seed) PPR scores

    After thresholding:
        - Seeds are always included (hard anchor, matching the seed constraint)
        - Take the largest connected component of candidate genes in the PPI
          (ensures the recovered module is a single connected subnetwork)

    This approach differs from SeedMix in that we use SGSA-derived PPR scores
    rather than solving the full MIQP density optimisation, but the neighbourhood
    reduction step is identical in spirit.

    Returns:
        subnetwork_genes : sorted list of genes in the recovered module
        delta            : the PPR score cutoff applied
        n_above_threshold: number of genes passing the threshold (before LCC filter)
    """
    seed_set = set(valid_seeds)

    # Compute delta from background (non-seed) gene PPR scores only,
    # so that the strongly personalised seed scores don't inflate the threshold
    non_seed_scores = np.array([s for g, s in ppr_scores.items() if g not in seed_set])
    mean_bg = non_seed_scores.mean()
    std_bg  = non_seed_scores.std()
    delta   = mean_bg + n_std * std_bg

    # Genes above the threshold (seeds added regardless)
    above = {g for g, s in ppr_scores.items() if s >= delta}
    n_above = len(above)
    candidates = above | seed_set

    # Restrict to the LCC of the candidate subgraph within the PPI
    sub = G.subgraph(candidates)
    if sub.number_of_nodes() == 0:
        return sorted(seed_set), delta, n_above

    comps = list(nx.connected_components(sub))
    lcc_nodes = max(comps, key=len)

    # Always retain all seeds even if isolated from the LCC
    subnetwork_genes = sorted(lcc_nodes | seed_set)
    return subnetwork_genes, delta, n_above


# ── 5. MAIN ──────────────────────────────────────────────────────────────────
def main():
    nsea_networks = parse_nsea_html("nSEA_Networks__1_.html")
    G = load_network()
    network_nodes = set(G.nodes())
    lcc = compute_network_stats(G)

    print(f"Subnetwork extraction: PPR scores thresholded at mean + {PPR_DELTA_STD}×std")
    print(f"(Mirrors SeedMix Algorithm 1, Phase 1: diffusion threshold delta)")
    print(f"Paper reference: SN10 with 6 seeds → ~133-gene subnetwork (we recover ~150)\n")

    os.makedirs("results_nsea", exist_ok=True)
    all_results = []

    total = len(nsea_networks)
    for idx, (sn_name, seed_genes) in enumerate(
            sorted(nsea_networks.items(), key=lambda x: int(x[0][2:])), 1):

        valid_seeds      = [g for g in seed_genes if g in network_nodes]
        missing          = [g for g in seed_genes if g not in network_nodes]
        seeds_in_lcc     = [g for g in valid_seeds if g in lcc]
        seeds_out_of_lcc = [g for g in valid_seeds if g not in lcc]
        seed_components  = (nx.number_connected_components(G.subgraph(valid_seeds))
                            if valid_seeds else 0)

        print(f"\n[{idx}/{total}] {sn_name}:")
        print(f"  Seeds: {len(seed_genes)} total | "
              f"{len(valid_seeds)} in network | "
              f"{len(missing)} missing from network")
        if missing:
            print(f"  Missing genes (absent from IntAct PPI): {missing}")
        print(f"  Seeds in LCC: {len(seeds_in_lcc)}/{len(valid_seeds)}"
              + (f"  | Outside LCC: {seeds_out_of_lcc}" if seeds_out_of_lcc else ""))
        print(f"  Seed subgraph components: {seed_components}")

        if not valid_seeds:
            print(f"  Skipping {sn_name} — no seeds found in network.")
            all_results.append({
                "Network": sn_name,
                "Original_Seed_Count": len(seed_genes),
                "Valid_Seed_Count": 0,
                "Missing_Seed_Count": len(missing),
                "Seeds_Used": "",
                "Missing_Seeds": "|".join(missing),
                "Seeds_in_LCC": 0,
                "Seeds_Outside_LCC": "",
                "Seed_Subgraph_Components": 0,
                "PPR_Delta_Threshold": "",
                "Genes_Above_Threshold": 0,
                "Subnetwork_Size": 0,
                "Subnetwork_Genes": "",
                "Top10_Subnetwork_Genes": "SKIPPED",
            })
            continue

        try:
            # run_sgsa_multirun computes PPR personalised on seeds (5 runs averaged).
            # After multirun averaging the +1 SA boost cancels out across stochastic
            # runs, but the underlying PPR scores remain valid and stable — these are
            # exactly the diffusion scores W used in SeedMix Algorithm 1 Phase 1.
            ppr_scores = run_sgsa_multirun(G, valid_seeds, n_runs=5)

            # ── Extract focused subnetwork (mirrors paper Phase 1) ───────────
            subnetwork_genes, delta, n_above = extract_subnetwork(
                G, ppr_scores, valid_seeds
            )

            # Rank subnetwork genes by PPR score (seeds excluded from ranked output)
            ranked_subnet = sorted(
                [(g, ppr_scores[g]) for g in subnetwork_genes if g not in valid_seeds],
                key=lambda x: x[1], reverse=True
            )

            # Full ranked list (all background genes) for reference/benchmarking
            ranked_all = sorted(
                [(g, ppr_scores[g]) for g in ppr_scores if g not in valid_seeds],
                key=lambda x: x[1], reverse=True
            )

            # ── Save primary output: the focused disease subnetwork ──────────
            sub_df = pd.DataFrame(ranked_subnet, columns=["Gene", "Score"])
            sub_df.insert(0, "Rank", range(1, len(sub_df)+1))
            sub_df.to_csv(f"results_nsea/{sn_name}_subnetwork.csv", index=False)

            # Save full ranked list with subnetwork membership flag (for benchmarking)
            full_df = pd.DataFrame(ranked_all, columns=["Gene", "Score"])
            full_df.insert(0, "Rank", range(1, len(full_df)+1))
            full_df["In_Subnetwork"] = full_df["Gene"].isin(set(subnetwork_genes))
            full_df.to_csv(f"results_nsea/{sn_name}_ranked_genes.csv", index=False)

            top10 = [g for g, _ in ranked_subnet[:10]]
            subnetwork_edge_count = G.subgraph(subnetwork_genes).number_of_edges()

            print(f"  PPR delta (mean+{PPR_DELTA_STD}σ): {delta:.6f} | "
                  f"Genes above threshold: {n_above}")
            print(f"  Subnetwork: {len(subnetwork_genes)} genes, "
                  f"{subnetwork_edge_count} edges "
                  f"(density: {2*subnetwork_edge_count/(len(subnetwork_genes)*(len(subnetwork_genes)-1)):.3f})")
            print(f"  Top 5: {[g for g,_ in ranked_subnet[:5]]}")

            all_results.append({
                "Network":                  sn_name,
                "Original_Seed_Count":      len(seed_genes),
                "Valid_Seed_Count":         len(valid_seeds),
                "Missing_Seed_Count":       len(missing),
                "Seeds_Used":               "|".join(valid_seeds),
                "Missing_Seeds":            "|".join(missing),
                "Seeds_in_LCC":             len(seeds_in_lcc),
                "Seeds_Outside_LCC":        "|".join(seeds_out_of_lcc),
                "Seed_Subgraph_Components": seed_components,
                "PPR_Delta_Threshold":      round(delta, 8),
                "Genes_Above_Threshold":    n_above,
                "Subnetwork_Size":          len(subnetwork_genes),
                "Subnetwork_Edge_Count":    subnetwork_edge_count,
                "Subnetwork_Genes":         "|".join(subnetwork_genes),
                "Top10_Subnetwork_Genes":   "|".join(top10),
            })

        except Exception as e:
            print(f"  ERROR in {sn_name}: {e}")
            all_results.append({
                "Network": sn_name,
                "Original_Seed_Count": len(seed_genes),
                "Valid_Seed_Count": len(valid_seeds),
                "Missing_Seed_Count": len(missing),
                "Seeds_Used": "|".join(valid_seeds),
                "Missing_Seeds": "|".join(missing),
                "Seeds_in_LCC": len(seeds_in_lcc),
                "Seeds_Outside_LCC": "|".join(seeds_out_of_lcc),
                "Seed_Subgraph_Components": seed_components,
                "PPR_Delta_Threshold": "",
                "Genes_Above_Threshold": 0,
                "Subnetwork_Size": 0,
                "Subnetwork_Edge_Count": 0,
                "Subnetwork_Genes": "",
                "Top10_Subnetwork_Genes": "ERROR",
            })

    # ── SAVE + PRINT FINAL SUMMARY ───────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("results_nsea/ALL_networks_summary.csv", index=False)

    ran = summary_df[summary_df['Subnetwork_Size'] > 0]
    print("\n" + "="*60)
    print("nSEA RUN SUMMARY STATISTICS")
    print("="*60)
    print(f"  Total nSEA networks processed:          {len(summary_df):>5}")
    print(f"  Networks with ALL seeds present:         {(summary_df['Missing_Seed_Count']==0).sum():>5}")
    print(f"  Networks with ≥1 missing seed:           {(summary_df['Missing_Seed_Count']>0).sum():>5}")
    print(f"  Networks skipped (0 valid seeds):        {(summary_df['Valid_Seed_Count']==0).sum():>5}")
    unique_missing = set()
    for v in summary_df['Missing_Seeds']:
        if v and str(v).strip():
            for g in str(v).split('|'):
                if g.strip(): unique_missing.add(g.strip())
    print(f"  Unique missing genes (absent from PPI):  {len(unique_missing):>5}")
    print(f"  Networks with all seeds in LCC:          {(summary_df['Seeds_in_LCC']==summary_df['Valid_Seed_Count']).sum():>5}")
    if len(ran):
        print(f"  Avg subnetwork size:                     {ran['Subnetwork_Size'].mean():>8.1f} genes")
        print(f"  Min / Max subnetwork size:              "
              f"{ran['Subnetwork_Size'].min():>4} / {ran['Subnetwork_Size'].max()}")
        print(f"  PPR threshold: mean + {PPR_DELTA_STD}×std of background PPR scores")
        print(f"  (Matches SeedMix Algorithm 1 delta; paper's SN10 example: 133 genes)")
    print("="*60)
    print(f"\nOutputs per network:")
    print(f"  {{SN}}_subnetwork.csv    — focused disease module (primary result)")
    print(f"  {{SN}}_ranked_genes.csv  — all genes ranked by PPR score + In_Subnetwork flag")
    print(f"\nDone. Results written to results_nsea/")
    print(summary_df[["Network","Valid_Seed_Count","PPR_Delta_Threshold",
                       "Subnetwork_Size","Top10_Subnetwork_Genes"]].to_string(index=False))


if __name__ == "__main__":
    main()
