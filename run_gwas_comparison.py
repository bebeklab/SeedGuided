#!/usr/bin/env python3
"""
run_gwas_comparison.py
=======================

A SELF-CONTAINED, HONEST comparison of SeedMix vs. the NetMix2 paper's
published Fig. 3b numbers, run on REAL GWAS summary statistics.

WHY THIS SCRIPT EXISTS (read this before running)
---------------------------------------------------
The existing benchmark harness (wrappers.py -> run_seedmix()) does NOT use
real GWAS p-values. It invents fake p-values for every gene, and it draws a
*different, stronger-signal* distribution for `hidden_seeds` (the genes you're
supposedly trying to discover) than for background genes. That leaks the
answer into the input and makes any AUC/PR number meaningless as a measure of
whether SeedMix actually works. It also is evaluating a different thing than
NetMix2's Fig. 3b, which reports the COUNT of curated reference disease genes
recovered inside the estimated altered subnetwork, given REAL GWAS gene-level
scores -- not a synthetic held-out AUC split.

This script instead:
  1. Reads your real, downloaded harmonised GWAS summary-statistics file
     (the *.h.tsv.gz you pulled from the GWAS Catalog FTP) for each disease.
  2. Maps SNPs -> genes using real genomic coordinates (you supply a gene
     coordinate file -- see --gene-coords below).
  3. Aggregates SNP p-values to one p-value per gene (min p-value in a window
     around the gene body -- a standard, simple gene-based aggregation).
  4. Defines seed genes as genes containing a genome-wide-significant hit
     (p < 5e-8) -- these are markers, not test labels; nothing about the
     *held-out* evaluation is fed into the algorithm.
  5. Runs the REAL run_seedmix3.py (not the wrappers.py mock) on those real
     scores. If Gurobi's license/token server is unreachable, it automatically
     falls back to a greedy local-search APPROXIMATION of the same objective
     (see GreedyFallback below) instead of failing outright -- every result
     row is labeled with which solver actually produced it, so approximate
     results are never mistaken for exact ones.
  6. Counts how many curated reference disease genes (Harmonizome DISEASES,
     the same source your run_benchmark.py already uses) land inside the
     estimated subnetwork -- the SAME metric NetMix2's Fig. 3b uses.
  7. Prints/saves a table next to the approximate values read off Fig. 3b so
     you can see a real, apples-to-apples comparison.

WHAT YOU NEED TO PROVIDE
-------------------------
--gwas NAME:/path/to/GCSTxxxxxxx.h.tsv.gz     (repeatable, one per disease)
--gene-coords /path/to/gene_coords.tsv         (see format below)
--reference-genes /path/to/gene_attribute_edges.txt.gz  (Harmonizome DISEASES file)
--network /path/to/PPI_2021_network.txt        (defaults to the repo copy)
--seedmix-dir /path/to/pipeline/seedmix        (defaults to ./pipeline/seedmix)
--allow-greedy-fallback                        (opt-in: use greedy approximation
                                                 if Gurobi's license/token server
                                                 is unreachable; OFF by default so
                                                 you never get approximate results
                                                 without explicitly asking for them)

GENE COORDINATE FILE FORMAT (--gene-coords)
--------------------------------------------
Tab or comma separated, with a header containing (any case, any order):
    gene    chrom    start    end
e.g.:
    gene    chrom   start      end
    TP53    17      7661779    7687538
Get this from UCSC Table Browser (refGene, hg38) or Ensembl BioMart
(GRCh38, columns: Gene name, Chromosome/scaffold name, Gene start (bp),
Gene end (bp)). Use GRCh38 coordinates -- GWAS Catalog harmonised files are
GRCh38. Gene symbols should match the PPI network's naming (HGNC symbols,
e.g. "TP53" not "ENSG00000141510").

EXAMPLE
-------
python run_gwas_comparison.py \\
    --gwas "Rheumatoid Arthritis:GWAS Data/RA/GCST90436650.h.tsv.gz" \\
    --gwas "Bipolar Disorder:GWAS Data/BD/GCSTxxxxxxx.h.tsv.gz" \\
    --gwas "Type 2 Diabetes:GWAS Data/T2D/GCSTxxxxxxx.h.tsv.gz" \\
    --gene-coords data/gene_coords_hg38.tsv \\
    --reference-genes "data/DISEASES Curated Gene-Disease Assocation Evidence Scores/gene_attribute_edges.txt.gz" \\
    --outdir results_gwas_comparison
"""

import argparse
import csv
import gzip
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

try:
    from intervaltree import IntervalTree
    HAVE_INTERVALTREE = True
except ImportError:
    HAVE_INTERVALTREE = False


# ==============================================================================
# Approximate values read directly off Fig. 3b of the NetMix2 paper.
# These are eyeballed from the bar chart image, NOT extracted from a table --
# treat them as approximate ground truth for a sanity check, and swap in exact
# numbers from the paper's supplementary tables if you have them.
# ==============================================================================
FIG3B_APPROX = {
    "Rheumatoid Arthritis": {"Scores Only": 17, "Network Propagation": 21, "NetMix2": 26, "Network Only": 15},
    "Bipolar Disorder":     {"Scores Only": 21, "Network Propagation": 46, "NetMix2": 53, "Network Only": 39},
    "Type 2 Diabetes":      {"Scores Only": 9,  "Network Propagation": 25, "NetMix2": 10, "Network Only": 20},
}

GENOME_WIDE_SIG = 5e-8


# ==============================================================================
# 1. Gene coordinates + interval index
# ==============================================================================
def load_gene_coords(path):
    sep = "\t" if path.endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        raise ValueError(f"Could not find a column among {cands} in {path}. "
                          f"Found columns: {list(df.columns)}")

    gene_col = pick("gene", "symbol", "gene_symbol", "hgnc_symbol", "name")
    chrom_col = pick("chrom", "chromosome", "chr")
    start_col = pick("start", "gene_start", "txstart")
    end_col = pick("end", "gene_end", "txend")

    out = df[[gene_col, chrom_col, start_col, end_col]].copy()
    out.columns = ["gene", "chrom", "start", "end"]
    out["chrom"] = out["chrom"].astype(str).str.replace("chr", "", case=False, regex=False).str.strip()
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["start", "end"])
    out = out[out["gene"].notna() & (out["gene"].astype(str).str.len() > 0)]
    return out


class GeneMapper:
    """Maps (chrom, pos) -> set of overlapping gene symbols, using a window
    padded around each gene body. Uses intervaltree if available (handles
    overlapping genes correctly); otherwise falls back to a nearest-single-gene
    approximation (documented simplification)."""

    def __init__(self, gene_coords_df, window=50_000):
        self.window = window
        self.use_tree = HAVE_INTERVALTREE
        self.trees = {}
        self.fallback = defaultdict(list)  # chrom -> sorted list of (start, end, gene)

        for chrom, sub in gene_coords_df.groupby("chrom"):
            if self.use_tree:
                t = IntervalTree()
                for _, row in sub.iterrows():
                    s = max(0, int(row["start"]) - window)
                    e = int(row["end"]) + window
                    if e > s:
                        t[s:e] = row["gene"]
                self.trees[chrom] = t
            else:
                items = sorted(
                    (max(0, int(r["start"]) - window), int(r["end"]) + window, r["gene"])
                    for _, r in sub.iterrows()
                )
                self.fallback[chrom] = items

        if not self.use_tree:
            print("[!] 'intervaltree' package not found -- using an approximate "
                  "nearest-gene fallback that may miss some overlapping genes. "
                  "For exact results run: pip install intervaltree --break-system-packages",
                  file=sys.stderr)

    def genes_at(self, chrom, pos):
        chrom = str(chrom).replace("chr", "").strip()
        if self.use_tree:
            t = self.trees.get(chrom)
            if not t:
                return []
            return [iv.data for iv in t[pos]]
        else:
            items = self.fallback.get(chrom)
            if not items:
                return []
            import bisect
            starts = [i[0] for i in items]
            idx = bisect.bisect_right(starts, pos) - 1
            hits = []
            # Small local scan around the insertion point (handles most overlaps)
            for i in range(max(0, idx - 5), min(len(items), idx + 6)):
                s, e, g = items[i]
                if s <= pos <= e:
                    hits.append(g)
            return hits


# ==============================================================================
# 2. Read harmonised GWAS file, map SNPs -> genes, aggregate to gene p-values
# ==============================================================================
def detect_columns(df_cols):
    cols = {c.lower(): c for c in df_cols}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    chrom = pick("hm_chrom", "chromosome", "chr", "#chrom", "#chr")
    pos = pick("hm_pos", "base_pair_location", "bp", "pos")
    pval = pick("p_value", "pvalue", "p", "pval")
    return chrom, pos, pval


def gwas_to_gene_pvalues(gwas_path, mapper, chunksize=500_000):
    """Streams a (possibly huge) harmonised GWAS file, maps each SNP to any
    genes whose padded window contains it, and keeps the MINIMUM p-value seen
    per gene (standard simple gene-based aggregation)."""
    gene_min_p = {}
    n_rows = 0
    n_mapped = 0

    opener = gzip.open if gwas_path.endswith(".gz") else open
    n_skip = 0
    header = None
    with opener(gwas_path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                n_skip += 1
                continue
            header = line.rstrip("\n").split("\t")
            break
    if header is None:
        raise ValueError(f"Could not find a header row in {gwas_path} -- every line started with '##'.")

    chrom_col, pos_col, pval_col = detect_columns(header)
    missing = [n for n, v in [("chromosome", chrom_col), ("position", pos_col), ("p_value", pval_col)] if v is None]
    if missing:
        raise ValueError(
            f"Could not find columns for {missing} in {gwas_path}. "
            f"File header was: {header}"
        )

    reader = pd.read_csv(
        gwas_path, sep="\t", chunksize=chunksize, skiprows=n_skip,
        usecols=[chrom_col, pos_col, pval_col],
        dtype={chrom_col: str},
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=[chrom_col, pos_col, pval_col])
        n_rows += len(chunk)
        for chrom, pos, pval in zip(chunk[chrom_col].values, chunk[pos_col].values, chunk[pval_col].values):
            try:
                pos = int(pos)
                pval = float(pval)
            except (ValueError, TypeError):
                continue
            genes = mapper.genes_at(chrom, pos)
            if not genes:
                continue
            n_mapped += 1
            for g in genes:
                cur = gene_min_p.get(g)
                if cur is None or pval < cur:
                    gene_min_p[g] = pval

    print(f"    -> read {n_rows:,} SNP rows, {n_mapped:,} mapped to >=1 gene, "
          f"{len(gene_min_p):,} distinct genes with a p-value.")
    return gene_min_p


# ==============================================================================
# 3. Load PPI network node set, curated reference disease genes
# ==============================================================================
def load_network_nodes(edge_list_path):
    nodes = set()
    with open(edge_list_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
    return nodes


def count_network_edges(edge_list_path):
    """Counts lines with 2+ tab-separated fields -- used to scale --num-edges
    as a fraction of the network's actual edge count, per NetMix2's own
    methodology (they used ~40% of |E| in the original network G, scaled
    per-network, rather than a fixed constant)."""
    n = 0
    with open(edge_list_path) as f:
        for line in f:
            if len(line.rstrip("\n").split("\t")) >= 2:
                n += 1
    return n


def load_reference_disease_genes(path, disease_name, override_label=None):
    # Columns are: source, source_desc, source_id, target, target_desc, target_id, weight.
    # A description row (skipped via skiprows=[1]) labels these as:
    # GeneSym, Ensembl Acc, GeneID, Disease, DOID, NA, weight -- so `target` (col 4) is
    # the actual disease NAME, and `target_desc` (col 5) is the DOID code, e.g.
    # "DOID:5241". (An earlier version of this function had these swapped -- fixed now.)
    dis_df = pd.read_csv(path, sep="\t", skiprows=[1], usecols=["source", "target"])
    dis_df.columns = ["Gene", "Disease"]
    groups = dis_df.groupby("Disease")["Gene"].apply(set).to_dict()

    if override_label is not None:
        if override_label in groups:
            return groups[override_label]
        lower_map = {k.lower(): k for k in groups}
        if override_label.lower() in lower_map:
            return groups[lower_map[override_label.lower()]]
        raise ValueError(
            f"--reference-label override '{override_label}' for '{disease_name}' was not "
            f"found in the reference gene file at all. Double-check the exact label with: "
            f"zcat '{path}' | cut -f2 | tail -n +3 | sort -u | grep -i <keyword>"
        )

    if disease_name in groups:
        return groups[disease_name]

    # Case-insensitive / substring fallback so small naming differences don't silently
    # return an empty set. NOTE: this only catches cases where the reference file's
    # vocabulary is close to plain-English disease names (e.g. "Rheumatoid Arthritis").
    # Some reference files use a different vocabulary entirely (e.g. rare/Mendelian
    # disease nomenclature) where no substring match will ever succeed -- in that case
    # you MUST use --reference-label to specify the exact label yourself.
    lower_map = {k.lower(): k for k in groups}
    if disease_name.lower() in lower_map:
        return groups[lower_map[disease_name.lower()]]

    candidates = [k for k in groups if disease_name.lower() in k.lower()]
    if len(candidates) == 1:
        print(f"    -> matched '{disease_name}' to reference disease label '{candidates[0]}'")
        return groups[candidates[0]]
    elif len(candidates) > 1:
        raise ValueError(
            f"Disease name '{disease_name}' matched multiple reference labels: {candidates}. "
            f"Use --reference-label \"{disease_name}:<exact label>\" to disambiguate."
        )
    else:
        raise ValueError(
            f"Disease name '{disease_name}' not found in reference gene file, and no "
            f"substring match exists either (this reference file may use a different "
            f"disease-naming vocabulary entirely). Find the correct label yourself with:\n"
            f"    zcat '{path}' | cut -f2 | tail -n +3 | sort -u | grep -i <keyword>\n"
            f"then rerun with: --reference-label \"{disease_name}:<exact label found above>\"\n"
            f"A few example labels in this file: {list(groups.keys())[:10]}"
        )


def load_reference_genes_gwas_catalog(path, trait_keyword, chunksize=200_000, exclude_compound=True):
    """Alternative reference gene source: the GWAS Catalog's own curated
    'all associations' file (columns include DISEASE/TRAIT and MAPPED_GENE).
    Filters rows whose DISEASE/TRAIT contains `trait_keyword` (case-insensitive
    substring), then parses MAPPED_GENE, which can contain:
      - multiple genes separated by ", "
      - intergenic entries like "GENE1 - GENE2" (SNP falls between two genes;
        we keep both flanking genes)

    By default (exclude_compound=True), rows whose DISEASE/TRAIT looks like a
    COMPOUND trait (cross-trait pleiotropy, MTAG, drug-interaction, or
    "X or Y"/"X vs Y" comparison studies) are excluded. Without this, a simple
    substring match pulls in studies where the target trait is only one of
    several unrelated conditions being studied together (e.g. "Peptic ulcer
    disease or bipolar disorder (pleiotropy)"), which inflates the reference
    gene set with genes that have nothing specifically to do with the target
    disease.
    """
    COMPOUND_MARKERS = [
        " or ", " vs ", " vs. ", " x ", "interaction", "pleiotropy",
        "mtag", "combined", "adjusted for", "shared genetic",
        "genetic correlation", "cross-disorder",
    ]

    def is_compound_trait(trait):
        t = trait.lower()
        return any(marker in t for marker in COMPOUND_MARKERS)

    genes = set()
    matched_traits = set()
    excluded_traits = set()
    n_matched_rows = 0

    exact_mode = trait_keyword.startswith("=")
    match_value = trait_keyword[1:].strip() if exact_mode else trait_keyword

    reader = pd.read_csv(
        path, sep="\t", chunksize=chunksize,
        usecols=["DISEASE/TRAIT", "MAPPED_GENE"],
        dtype=str, low_memory=False, on_bad_lines="skip",
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["DISEASE/TRAIT"])
        if exact_mode:
            mask = chunk["DISEASE/TRAIT"].str.lower().str.strip() == match_value.lower()
        else:
            mask = chunk["DISEASE/TRAIT"].str.contains(match_value, case=False, na=False)
        hits = chunk[mask]
        if hits.empty:
            continue

        if exclude_compound and not exact_mode:
            compound_mask = hits["DISEASE/TRAIT"].apply(is_compound_trait)
            excluded_traits.update(hits.loc[compound_mask, "DISEASE/TRAIT"].unique())
            hits = hits[~compound_mask]
            if hits.empty:
                continue

        n_matched_rows += len(hits)
        matched_traits.update(hits["DISEASE/TRAIT"].unique())
        for mapped in hits["MAPPED_GENE"].dropna():
            for part in mapped.split(","):
                for g in part.split(" - "):
                    g = g.strip()
                    if g:
                        genes.add(g)

    if not genes:
        raise ValueError(
            f"No rows in the GWAS Catalog file matched trait keyword '{trait_keyword}' "
            f"after excluding compound/pleiotropy traits. Double-check the keyword against "
            f"actual DISEASE/TRAIT values with:\n"
            f"    cut -f8 '{path}' | tail -n +2 | sort -u | grep -i <keyword>\n"
            f"Or pass exclude_compound=False if you want the broader (less clean) match."
        )

    print(f"    -> GWAS Catalog: '{trait_keyword}' matched {n_matched_rows} rows across "
          f"{len(matched_traits)} distinct trait labels, {len(genes)} distinct mapped genes.")
    print(f"       Matched trait labels: {sorted(matched_traits)}")
    if excluded_traits:
        print(f"       [!] Excluded {len(excluded_traits)} compound/pleiotropy trait labels "
              f"(these mention '{trait_keyword}' but study it jointly with unrelated traits): "
              f"{sorted(excluded_traits)[:10]}{' ...' if len(excluded_traits) > 10 else ''}")
    return genes


# ==============================================================================
# 4b. GREEDY FALLBACK -- only used if Gurobi's license/token server is down
#     and --allow-greedy-fallback was passed. This is an APPROXIMATION, not
#     the real SeedMix ILP solution. It maximizes the same kind of objective
#     (sum of gene z-scores in the subnetwork, seeds forced in, roughly
#     respecting the target density) via greedy growth + local-search swaps,
#     with no solver dependency at all. Results from this path are always
#     labeled "Greedy (approximate)" downstream -- never conflated with a
#     real Gurobi-solved SeedMix result.
# ==============================================================================
GUROBI_FAILURE_SIGNATURES = (
    "GurobiError", "token server", "No route to host",
    "license", "TokenServer", "Failed to connect",
)


def is_gurobi_connectivity_failure(stderr_text):
    return any(sig.lower() in stderr_text.lower() for sig in GUROBI_FAILURE_SIGNATURES)


def greedy_fallback_subnetwork(node_list, A_network, resps, seed_indices, alpha, rho,
                                max_iter=4000, rng_seed=0):
    """Greedy + local-search approximation of the netmix_edgedense objective:
    maximize sum(resps[S]) subject to seeds in S, |S| <= alpha * n, and
    (approximately) an edge-density target of rho on the induced subgraph.

    This does NOT guarantee optimality or exact density compliance the way
    the Gurobi ILP does -- it's a best-effort stand-in so you have SOME
    result while the license/network issue gets resolved.
    """
    rng = random.Random(rng_seed)
    n = len(node_list)
    G = nx.from_numpy_array(A_network)

    cap = max(int(round(alpha * n)), len(seed_indices))
    S = set(seed_indices)

    def neighbors_of(members):
        nbrs = set()
        for u in members:
            nbrs.update(G.neighbors(u))
        return nbrs - members

    def density(members):
        if len(members) < 2:
            return 1.0
        sub = G.subgraph(members)
        possible = len(members) * (len(members) - 1) / 2
        return sub.number_of_edges() / possible if possible > 0 else 0.0

    def objective(members):
        return sum(resps[i] for i in members)

    # 1. Greedy growth: repeatedly add the highest-scoring neighbor of S
    while len(S) < cap:
        candidates = neighbors_of(S)
        if not candidates:
            break
        best = max(candidates, key=lambda i: resps[i])
        S.add(best)

    # 2. Local search: swap in higher-scoring neighbors for the weakest
    #    non-seed members, as long as density doesn't collapse.
    cur_obj = objective(S)
    seed_set = set(seed_indices)
    for _ in range(max_iter):
        candidates = list(neighbors_of(S))
        removable = list(S - seed_set)
        if not candidates or not removable:
            break
        add_node = rng.choice(candidates)
        weakest = min(removable, key=lambda i: resps[i])
        if resps[add_node] <= resps[weakest]:
            continue
        proposal = (S - {weakest}) | {add_node}
        # Relaxed density tolerance -- greedy search can't hit an exact
        # ILP-quality density constraint, so we allow some slack.
        if density(proposal) + 1e-9 >= rho * 0.5:
            new_obj = objective(proposal)
            if new_obj > cur_obj:
                S, cur_obj = proposal, new_obj

    return sorted(S)


def write_seedmix_inputs(outdir, disease_key, network_nodes, gene_min_p):
    os.makedirs(outdir, exist_ok=True)
    score_path = os.path.join(outdir, "gene_scores.tsv")
    seed_path = os.path.join(outdir, "seed_genes.txt")

    seeds = []
    with open(score_path, "w") as f:
        for gene in sorted(network_nodes):
            # Genes with no GWAS-mapped SNP get p=1.0 (non-significant background),
            # NOT p=0 -- p=0 would cause run_seedmix3.py to silently drop the gene
            # from the network entirely (see src/common.py restrict_to_genes_in_network).
            p = gene_min_p.get(gene, 1.0)
            p = min(max(p, 1e-300), 1.0)
            f.write(f"{gene}\t{p}\n")
            if p < GENOME_WIDE_SIG:
                seeds.append(gene)

    with open(seed_path, "w") as f:
        for g in seeds:
            f.write(f"{g}\n")

    print(f"    -> {len(seeds)} genome-wide-significant (p<{GENOME_WIDE_SIG:g}) seed genes")
    return score_path, seed_path, seeds


def run_seedmix3(seedmix_dir, edge_list, gene_score, seed_genes, out_dir,
                  density=0.05, num_edges=175000, time_limit=0.5, thread_count=8, verbose=1,
                  allow_greedy_fallback=False):
    """Returns (subnetwork_gene_list, solver_used) where solver_used is
    'Gurobi (exact)' or 'Greedy (approximate)'. Never silently mixes the two --
    the caller must record solver_used alongside every result."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "run_seedmix3.py",
        "--edge_list", os.path.abspath(edge_list),
        "--gene_score", os.path.abspath(gene_score),
        "--seed_genes", os.path.abspath(seed_genes),
        "--density", str(density),
        "--num_edges", str(num_edges),
        "--time_limit", str(time_limit),
        "--thread_count", str(thread_count),
        "--edge_dense_linear",
        "--verbose", str(verbose),
        "-o", os.path.abspath(out_dir),
    ]
    print(f"    -> running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=seedmix_dir, capture_output=True, text=True, timeout=3600 * (time_limit + 0.5))

    if result.returncode == 0:
        subnet_file = os.path.join(out_dir, "netmix_subnetwork.tsv")
        if os.path.exists(subnet_file):
            with open(subnet_file) as f:
                genes = [line.strip() for line in f if line.strip()]
            return genes, "Gurobi (exact)"
        print("    [!] Gurobi run exited cleanly but produced no output file.")
        return [], "Gurobi (exact)"

    # Gurobi run failed -- figure out why before deciding what to do.
    combined_err = (result.stdout or "") + (result.stderr or "")
    print("    [!] run_seedmix3.py FAILED")
    print(result.stdout[-2000:])
    print(result.stderr[-2000:])

    if not is_gurobi_connectivity_failure(combined_err):
        print("    [!] Failure does not look like a Gurobi license/connectivity issue -- "
              "not falling back, this needs to be fixed directly.")
        return [], "Gurobi (exact) -- FAILED, no result"

    if not allow_greedy_fallback:
        print("    [!] This looks like a Gurobi license/token-server connectivity failure. "
              "Re-run with --allow-greedy-fallback if you want an APPROXIMATE result while "
              "that gets fixed. No fallback was used -- returning no result.")
        return [], "Gurobi (exact) -- FAILED (license/network), no fallback used"

    print("    [!] Gurobi unreachable -- falling back to greedy APPROXIMATION "
          "(this is NOT the exact SeedMix ILP result).")
    return _run_greedy_fallback(seedmix_dir, edge_list, gene_score, seed_genes,
                                 density=density, num_edges=num_edges), "Greedy (approximate)"


def _run_greedy_fallback(seedmix_dir, edge_list, gene_score, seed_genes, density, num_edges):
    """Reuses the real pipeline's own data-loading and z-score code (no Gurobi
    involved in any of it) so the fallback sees exactly the same processed
    network and scores the real solver would have seen -- only the final
    optimization step differs."""
    sys.path.insert(0, seedmix_dir)
    try:
        from src.common import (
            read_genes_from_file, load_network, load_pvalues,
            restrict_to_genes_in_network, compute_zscores, post_process_zscores,
        )
    finally:
        sys.path.pop(0)

    seed_names = read_genes_from_file(seed_genes) or []
    node_list, A_network = load_network(edge_list, verbosity=0)
    if not node_list:
        return []

    pvals = load_pvalues(gene_score, node_list, verbosity=0)
    pvals, node_list, A_network = restrict_to_genes_in_network(pvals, node_list, A_network, verbosity=0)
    node_list = list(node_list)
    if not node_list:
        return []

    zscores = compute_zscores(pvals)
    resps = post_process_zscores(zscores)

    node_to_idx = {g: i for i, g in enumerate(node_list)}
    seed_indices = [node_to_idx[g] for g in seed_names if g in node_to_idx]
    if not seed_indices:
        print("    [!] None of the seed genes survived filtering -- greedy fallback cannot run.")
        return []

    target_size = max(len(seed_indices) * 2, 15)
    alpha = min(target_size / max(len(node_list), 1), 0.95)

    subnet_indices = greedy_fallback_subnetwork(node_list, A_network, resps, seed_indices, alpha, density)
    return [node_list[i] for i in subnet_indices]


# ==============================================================================
# 5. Main
# ==============================================================================
def parse_gwas_arg(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError("--gwas must be NAME:/path/to/file, e.g. 'Bipolar Disorder:GWAS Data/bd.h.tsv.gz'")
    name, path = s.split(":", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gwas", action="append", type=parse_gwas_arg, required=True,
                     help="NAME:/path/to/harmonised.h.tsv.gz  (repeatable)")
    ap.add_argument("--gene-coords", required=True, help="Gene coordinate TSV (see docstring for format)")
    ap.add_argument("--reference-genes", required=True,
                     help="Harmonizome DISEASES gene_attribute_edges.txt.gz (curated reference gene sets)")
    ap.add_argument("--reference-label", action="append", default=[], type=parse_gwas_arg,
                     help="NAME:EXACT_REFERENCE_LABEL (repeatable). Use this when the reference "
                          "file's disease-naming vocabulary doesn't match your --gwas NAME at all "
                          "(e.g. it uses rare-disease nomenclature). Find exact labels with: "
                          "zcat <reference-genes file> | cut -f2 | tail -n +3 | sort -u | grep -i <keyword>")
    ap.add_argument("--gwas-catalog-file", default=None,
                     help="Optional path to the GWAS Catalog 'all associations' TSV "
                          "(gwas-catalog-download-associations-v1.0-full.tsv). If a disease has a "
                          "--gwas-catalog-trait entry, its reference genes come from THIS file "
                          "instead of --reference-genes.")
    ap.add_argument("--gwas-catalog-trait", action="append", default=[], type=parse_gwas_arg,
                     help="NAME:TRAIT_KEYWORD (repeatable). Pulls reference genes for that disease "
                          "from the GWAS Catalog file's MAPPED_GENE column, for rows whose "
                          "DISEASE/TRAIT contains TRAIT_KEYWORD (case-insensitive substring). "
                          "Requires --gwas-catalog-file to be set.")
    ap.add_argument("--network", default="pipeline/seedmix/data/PPI_2021_network.txt")
    ap.add_argument("--seedmix-dir", default="pipeline/seedmix")
    ap.add_argument("--outdir", default="results_gwas_comparison")
    ap.add_argument("--window", type=int, default=50_000, help="Padding (bp) around each gene body for SNP mapping")
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--num-edges", type=int, default=175000)
    ap.add_argument("--num-edges-fraction", type=float, default=None,
                     help="If set, overrides --num-edges by computing "
                          "round(fraction * total_edges_in_network) instead. The NetMix2 paper "
                          "used ~0.4 (40%% of the network's edges) for its similarity threshold "
                          "graph, scaled per-network rather than a fixed constant. Recommended "
                          "over a fixed --num-edges when comparing against paper-reported numbers.")
    ap.add_argument("--time-limit", type=float, default=0.5, help="Gurobi time limit in hours, per disease")
    ap.add_argument("--thread-count", type=int, default=8)
    ap.add_argument("--allow-greedy-fallback", action="store_true",
                     help="If Gurobi's license/token server is unreachable, fall back to a "
                          "greedy APPROXIMATION instead of failing. OFF by default -- results "
                          "produced this way are clearly labeled 'Greedy (approximate)' in the "
                          "output and should not be treated as the exact SeedMix result.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    reference_label_overrides = dict(args.reference_label)
    gwas_catalog_traits = dict(args.gwas_catalog_trait)

    print("Loading gene coordinates...")
    gene_coords = load_gene_coords(args.gene_coords)
    print(f"  -> {len(gene_coords)} gene coordinate records")
    mapper = GeneMapper(gene_coords, window=args.window)

    print("Loading PPI network node set...")
    network_nodes = load_network_nodes(args.network)
    print(f"  -> {len(network_nodes)} genes in network")

    effective_num_edges = args.num_edges
    if args.num_edges_fraction is not None:
        total_edges = count_network_edges(args.network)
        effective_num_edges = round(args.num_edges_fraction * total_edges)
        print(f"  -> --num-edges-fraction {args.num_edges_fraction} * {total_edges} network edges "
              f"= {effective_num_edges} (overrides --num-edges)")

    rows = []
    for disease_name, gwas_path in args.gwas:
        print(f"\n=== {disease_name} ===")
        print(f"  Reading GWAS file: {gwas_path}")
        gene_min_p = gwas_to_gene_pvalues(gwas_path, mapper)

        disease_outdir = os.path.join(args.outdir, disease_name.replace(" ", "_"))
        score_path, seed_path, seeds = write_seedmix_inputs(disease_outdir, disease_name, network_nodes, gene_min_p)

        if not seeds:
            print(f"  [!] No genome-wide-significant genes found for {disease_name}. "
                  f"Check --window, your gene-coords build (must be GRCh38), and the GWAS file's p-value column.")
            rows.append({"Disease": disease_name, "Solver": "-", "SeedMix_RefGenes": 0, "SubnetworkSize": 0})
            continue

        subnet_dir = os.path.join(disease_outdir, "seedmix_output")
        subnetwork_genes, solver_used = run_seedmix3(
            args.seedmix_dir, args.network, score_path, seed_path, subnet_dir,
            density=args.density, num_edges=effective_num_edges,
            time_limit=args.time_limit, thread_count=args.thread_count,
            allow_greedy_fallback=args.allow_greedy_fallback,
        )
        print(f"  -> Solver used: {solver_used}")
        print(f"  -> SeedMix estimated subnetwork size: {len(subnetwork_genes)}")

        if disease_name in gwas_catalog_traits:
            if not args.gwas_catalog_file:
                raise ValueError(
                    f"--gwas-catalog-trait was set for '{disease_name}' but --gwas-catalog-file "
                    f"was never provided."
                )
            ref_genes = load_reference_genes_gwas_catalog(
                args.gwas_catalog_file, gwas_catalog_traits[disease_name]
            )
        else:
            ref_genes = load_reference_disease_genes(
                args.reference_genes, disease_name,
                override_label=reference_label_overrides.get(disease_name),
            )
        recovered = set(subnetwork_genes) & ref_genes
        print(f"  -> {len(recovered)} / {len(ref_genes)} curated reference genes recovered")

        rows.append({
            "Disease": disease_name,
            "Solver": solver_used,
            "SeedMix_RefGenes": len(recovered),
            "SubnetworkSize": len(subnetwork_genes),
            "NumSeeds": len(seeds),
            "NumReferenceGenes": len(ref_genes),
            "ReferenceSource": "GWAS Catalog" if disease_name in gwas_catalog_traits else "Harmonizome DISEASES",
        })

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(args.outdir, "seedmix_vs_netmix2_comparison.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n" + "=" * 70)
    print("COMPARISON: reference genes recovered in estimated subnetwork")
    print("(NetMix2 / Network Propagation / Scores Only / Network Only columns")
    print(" are approximate values read off Fig. 3b of the NetMix2 paper --")
    print(" not independently re-run here.)")
    print("=" * 70)
    header = f"{'Disease':<22}{'Solver':<28}{'SeedMix':>10}{'NetMix2':>10}{'NetProp':>10}{'ScoresOnly':>12}{'NetOnly':>10}"
    print(header)
    any_approx = False
    for _, r in results_df.iterrows():
        fig = FIG3B_APPROX.get(r["Disease"], {})
        solver = r.get("Solver", "-")
        if "approximate" in str(solver).lower():
            any_approx = True
        print(f"{r['Disease']:<22}{str(solver):<28}{r['SeedMix_RefGenes']:>10}"
              f"{fig.get('NetMix2', '-'):>10}{fig.get('Network Propagation', '-'):>10}"
              f"{fig.get('Scores Only', '-'):>12}{fig.get('Network Only', '-'):>10}")

    if any_approx:
        print("\n[!] One or more rows used the greedy APPROXIMATE solver (Gurobi was "
              "unreachable). These are NOT exact SeedMix results -- rerun with a working "
              "Gurobi license/token-server connection before treating any comparison to "
              "NetMix2 as final.")

    print(f"\nFull results written to: {results_csv}")


if __name__ == "__main__":
    main()
