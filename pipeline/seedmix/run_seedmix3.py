import argparse
import csv
import pandas as pd
import numpy as np
from scipy.stats import norm
import networkx as nx
import sys
import os

# Import functions from your common.py file
from src.common import (
    read_genes_from_file, write_list_to_file, load_network,
    load_pvalues, restrict_to_genes_in_network, compute_zscores,
    post_process_zscores, correct_nans_from_locfdr, compute_ppr_kernel
)

# Import netmix functions from your netmix3.py file
from src.netmix3 import netmix_edgedense, netmix_cut # Assuming both are in netmix3.py


# Parse arguments
def get_parser():
    """
    Sets up the argument parser for the NetMix script.
    """
    description = 'Run NetMix analysis on a biological network with gene scores, optionally using seed genes.'
    parser = argparse.ArgumentParser(description=description, fromfile_prefix_chars='@')
    parser.add_argument('-el', '--edge_list', required=True, type=str, help='Path to the network edge list file (tab-separated).')
    parser.add_argument('-gs', '--gene_score', required=True, type=str, help='Path to the gene-to-score file (tab-separated, gene\\tscore).')
    # Argument for seed genes file
    parser.add_argument('-seeds', '--seed_genes', required=False, type=str,
                        help='Optional: Path to a file listing seed gene names (one gene per line).')
    parser.add_argument('-d', '--delta', required=False, type=float,
                        help='Delta threshold for constructing G_delta from PPR similarity. If not provided, uses --num_edges.')
    parser.add_argument('-num_edges', '--num_edges', required=False, type=int, default=175000,
                        help='Number of top edges to keep in G_delta if --delta is not provided.')
    parser.add_argument('-density', '--density', required=False, type=float, default=0.05,
                        help='Target edge density for the output subnetwork (used in the quadratic constraint).')
    parser.add_argument('-time_limit', '--time_limit', required=False, type=float, default=.5,
                        help='Time limit (hours) for the Gurobi solver in the netmix function.')
    parser.add_argument('-v', '--verbose', required=False, type=int, default=0,
                        help='Print the progress and details of running NetMix (0: silent, 1: info, 2: debug).')
    parser.add_argument('-o', '--output', required=False, type=str,
                        help='Directory for writing netmix results (e.g., subnetwork gene list).')
    # Add arguments for NetMix method and parameters if needed (currently hardcoded to edgedense)
    # parser.add_argument('--method_type', required=False, type=str, default='edgedense', choices=['edgedense', 'cut'], help='NetMix method to use.')
    # parser.add_argument('--edge_dense_quad', action='store_true', help='Use quadratic edge density constraint (default for edgedense).')
    # parser.add_argument('--edge_dense_linear', action='store_true', help='Use linear edge density constraint (requires Gurobi 9+).')
    parser.add_argument('--edge_dense_quad', action='store_true', help='Use quadratic edge density constraint (default for edgedense).')
    parser.add_argument('--edge_dense_linear', action='store_true', help='Use linear edge density constraint (requires Gurobi 9+).')
    # parser.add_argument('--gurobi_method', required=False, type=int, help='Gurobi Method parameter.')
    parser.add_argument('--thread_count', required=False, type=int, help='Gurobi Threads parameter.')
    # parser.add_argument('--mipgap', required=False, type=float, help='Gurobi MIPGap parameter.')


    return parser


def run(args):
    """
    Main function to load data, run NetMix with optional seed genes, and output results.
    """
    # Set verbosity level for helper functions
    verbosity = args.verbose

    ###########################################################
    # 1. Read seed gene names from file (if provided)
    seed_gene_names = None
    if args.seed_genes:
        seed_gene_names = read_genes_from_file(args.seed_genes)
        if seed_gene_names is None:
            print("Exiting due to error reading seed genes file.")
            sys.exit(1)
        if verbosity > 0:
            print(f"Read {len(seed_gene_names)} seed gene names from {args.seed_genes}.")
            if verbosity > 1:
                 print(f"Seed gene names: {seed_gene_names}")
        if not seed_gene_names and args.seed_genes: # Check if file was provided but empty
             print("Warning: Seed genes file was provided but is empty.")


    ###########################################################
    # 2. Load network
    node_list, A_network = load_network(args.edge_list, verbosity)
    if node_list is None or A_network is None:
        print("Exiting due to error loading network.")
        sys.exit(1)
    if len(node_list) == 0:
        print("Loaded an empty network. Exiting.")
        # Optionally write empty output files
        if args.output:
             os.makedirs(args.output, exist_ok=True)
             write_list_to_file(os.path.join(args.output, 'netmix_subnetwork.tsv'), [])
             write_list_to_file(os.path.join(args.output, 'node_list.tsv'), [])
        sys.exit(0)


    ###########################################################
    # 3. Load p-values for each gene
    pvals_list = load_pvalues(args.gene_score, node_list, verbosity)
    if pvals_list is None:
        print("Exiting due to error loading p-values.")
        sys.exit(1)

    ###########################################################
    # 4. Restrict to genes in the network with p-values > 0 and the largest connected component
    # This step updates the node list and adjacency matrix, changing the indices.
    processed_pvals, processed_node_list, processed_A = restrict_to_genes_in_network(
        pvals_list, node_list, A_network, verbosity
    )

    if processed_node_list is None or len(processed_node_list) == 0:
        print("No valid network or nodes remaining after filtering. Exiting.")
        # Optionally write an empty output file or handle this case as needed
        if args.output:
             os.makedirs(args.output, exist_ok=True)
             write_list_to_file(os.path.join(args.output, 'netmix_subnetwork.tsv'), [])
             write_list_to_file(os.path.join(args.output, 'node_list.tsv'), [])
        sys.exit(0) # Exit cleanly if no nodes found

    # Update node_list and A_network to the processed versions for subsequent steps
    node_list = processed_node_list
    A_network = processed_A
    pvals_list = processed_pvals # Keep pvals_list updated too

    ###########################################################
    # 5. Compute zscores (used as responses for NetMix)
    zscores = compute_zscores(pvals_list)
    resps = post_process_zscores(zscores, verbosity) # Use post-processed z-scores as responses

    ###########################################################
    # 6. Fit local fdr (if needed for other purposes, not directly used in netmix objective here)
    # If you intend to use locfdr output as responses, you would modify step 5
    # and potentially use correct_nans_from_locfdr.
    # For now, keeping this part as it was in your original script.
    try:
        # Check if there are enough data points for locfdr
        if len(zscores) >= 10: # locfdr typically needs at least 10 points
            r_locfdr=locfdr(zscores, nulltype=1, plot=0)
            # correct the nans in the locfdr (if you plan to use its output)
            # nonnull_count = correct_nans_from_locfdr(r_locfdr, zscores, "mlest", verbosity)
            # If you need the nonnull_count for alpha calculation based on locfdr, uncomment above
            # and potentially adjust alpha calculation below.
            if verbosity > 0:
                 print("locfdr fit completed.")
        else:
            if verbosity > 0:
                 print(f"Skipping locfdr fit: Insufficient data points ({len(zscores)}).")

    except Exception as e:
         print(f"Warning: Could not fit locfdr. Error: {e}")
         # Handle cases where locfdr might fail (e.g., insufficient data points)
         # If locfdr is essential for your alpha calculation, you might need to exit or use a fallback.


    ##################################################
    # 7. Compute PPR matrix (used for G_delta)
    PPR_sim_mat, PPR_mat_rowsums = compute_ppr_kernel(A_network, verbosity)
    if PPR_sim_mat is None or len(PPR_sim_mat) == 0:
        print("Exiting due to error computing PPR kernel or empty result.")
        # Optionally write empty output files
        if args.output:
             os.makedirs(args.output, exist_ok=True)
             write_list_to_file(os.path.join(args.output, 'netmix_subnetwork.tsv'), [])
             write_list_to_file(os.path.join(args.output, 'node_list.tsv'), [])
        sys.exit(1)

    ###########################################################
    # 8. Compute G_delta by thresholding PPR similarity matrix
    sim_mat = PPR_sim_mat

    # Remove diagonal (self-similarity)
    sim_mat_nodiag = sim_mat - np.diag(np.diag(sim_mat))

    # Determine delta threshold
    if args.delta is not None:
        delta = args.delta
        if verbosity > 0:
             print(f"Using specified delta: {delta}")
    else:
        # Sort non-diagonal upper triangle elements in descending order
        upper_triangle_indices = np.triu_indices(sim_mat_nodiag.shape[0], 1)
        if len(upper_triangle_indices[0]) == 0: # No edges in the similarity matrix
             print("Warning: No edges in the PPR similarity matrix to threshold. Delta set to 0.")
             delta = 0
             sim_mat_nodiag_sorted = np.array([])
        else:
            sim_mat_nodiag_sorted = np.sort(sim_mat_nodiag[upper_triangle_indices])[::-1]

        # Ensure args.num_edges is within bounds
        if args.num_edges >= len(sim_mat_nodiag_sorted):
             if verbosity > 0:
                print(f"Warning: Requested num_edges ({args.num_edges}) is >= total possible edges ({len(sim_mat_nodiag_sorted)}). Using all edges.")
             delta = sim_mat_nodiag_sorted[-1] if len(sim_mat_nodiag_sorted) > 0 else 0
        else:
            delta = sim_mat_nodiag_sorted[args.num_edges]
            if verbosity > 0:
                 print(f"Using delta based on {args.num_edges} edges: {delta}")


    # Create G_delta (binary adjacency matrix)
    sim_mat_delta = (sim_mat_nodiag > delta).astype(int) # Ensure binary (0 or 1)

    num_edges_original_network = int(np.sum(A_network)/2) # Edges in the processed network
    num_edges_delta = int(np.sum(sim_mat_delta)/2) # Edges in G_delta

    if verbosity > 0:
        print(f"Number of nodes in processed network: {len(node_list)}")
        print(f"Number of edges in processed network (LCC): {num_edges_original_network}")
        print(f"Delta threshold used for G_delta: {delta}")
        print(f"Number of edges in G_delta: {num_edges_delta}")

    # Check if G_delta is empty
    if num_edges_delta == 0 and len(node_list) > 1:
        print("Warning: G_delta has no edges. NetMix density constraint might not be effective.")
    if len(node_list) <= 1:
         print("Warning: Processed network has 0 or 1 node. NetMix might not be meaningful.")


    ###########################################################
    # 9. Parameters for netmix
    # Alpha determines the maximum size of the subnetwork relative to the processed network size.
    # The netmix function will ensure the size is at least the number of seed nodes.
    alpha = args.density # Assuming 'density' argument is used for alpha here, as in your original code snippet
                         # If 'density' was intended for rho, you might need a separate alpha argument.
                         # Let's assume args.density is the alpha value for clique size.
                         # The rho parameter for the density constraint is handled below.

    # Calculate rho for the density constraint. This should be the target density value.
    target_edge_density_value = 0.05 # Assuming a default target density if not specified elsewhere
                                     # If args.density was meant for target density, use that here.
                                     # Let's use a separate variable for clarity.
    # Based on your original code `rho=target_edge_density*(s-1)`, where `s` was nonnull_count,
    # it seems `rho` was *not* the target density itself, but a value derived from it
    # and the *expected* size. This is confusing.
    # The Gurobi constraint `LHS >= rho * 0.5*( vecsum*vecsum - vecsum )` implies `rho` *is* the target density.
    # Let's assume `args.density` *is* the target density you want to enforce in the constraint.
    target_edge_density_value = args.density # Use the value from the --density argument

    rho_for_netmix = target_edge_density_value # Pass the target density value directly

    if verbosity > 0:
        print(f'Alpha parameter for clique size: {alpha}')
        print(f'Target edge density constraint value (rho for NetMix): {rho_for_netmix}')


    ###########################################################
    # 10. Map seed gene names to indices in the *processed* network
    # This was moved up to ensure we use indices from the processed network.
    # processed_node_to_index mapping is available from step 4.
    seed_node_indices = []
    if seed_gene_names: # Only process if seed names were loaded
        processed_node_to_index = {node: i for i, node in enumerate(node_list)} # Use updated node_list
        found_seed_count = 0
        for gene_name in seed_gene_names:
            if gene_name in processed_node_to_index:
                seed_node_indices.append(processed_node_to_index[gene_name])
                found_seed_count += 1
            else:
                if verbosity > 0:
                     print(f"Warning: Seed gene '{gene_name}' not found in the processed network. Skipping.")

        if verbosity > 0:
             print(f"Found {found_seed_count} out of {len(seed_gene_names)} seed genes in the processed network.")
             if found_seed_count == 0 and len(seed_gene_names) > 0:
                  print("Warning: None of the provided seed genes were found in the network with p-values > 0 or in the largest connected component.")
    else:
        if verbosity > 0:
            print("No seed genes file provided. Running NetMix without forced seeds.")
        seed_node_indices = [] # Ensure it's an empty list if no seeds

    # Check if the number of seed nodes exceeds the alpha-based initial clique size
    initial_clique_size_alpha = int(len(node_list) * alpha)
    if len(seed_node_indices) > initial_clique_size_alpha:
        if verbosity > 0:
             print(f"Warning: Number of seed nodes ({len(seed_node_indices)}) exceeds alpha-based initial clique size ({initial_clique_size_alpha}). Effective clique size will be adjusted.")
        # The netmix function handles the actual adjustment, but this print is informative.


    ###########################################################
    # 11. RUN NETMIX
    if verbosity > 0:
        print('running netmix_edgedense')
        print("time_limit (hours):", args.time_limit)
        gurobi_output = True
    else:
        gurobi_output = False

    # Pass the seed_nodes list to the netmix function
    # Assuming edge_dense_quad is the default based on your original code structure
    # If you need to control quad/linear via args, uncomment related parser args
    # and pass them here.
    est_subnetwork_indices = netmix_edgedense(
        sim_mat_delta, # Use G_delta as the network for optimization
        rho_for_netmix, # Pass the target density value
        resps,
        seed_nodes=seed_node_indices, # Pass the mapped seed indices
        alpha=alpha, # Pass alpha (used for initial clique size calculation)
        edge_dense_quad=args.edge_dense_quad,      # <-- Uses the command line
        edge_dense_linear=args.edge_dense_linear,  # <-- Uses the command line       
        output=gurobi_output,
        time_limit=3600*args.time_limit, # Convert hours to seconds
        method=None, # Pass Gurobi method if added to args
        thread_count=None, # Pass Gurobi threads if added to args
        mipgap=None # Pass Gurobi mipgap if added to args
    )

    if est_subnetwork_indices is None: # Handle potential None return from netmix
        print("NetMix optimization failed or returned None.")
        est_subnetwork_indices = [] # Ensure it's an empty list on failure


    ###########################################################
    # 12. print solution summary
    solution_size = len(est_subnetwork_indices)
    # Map indices back to gene names using the *processed* node_list
    est_subnetwork_genes = [node_list[i] for i in est_subnetwork_indices] if solution_size > 0 else []

    # Calculate density based on the *original* processed network (A_network), not G_delta
    # G_delta is used for the optimization constraint, but the output subnetwork's
    # density is typically reported on the original network structure.
    solution_network_on_original_A = None
    num_edges_in_solution_on_original_A = 0
    solution_network_density_on_original_A = 0

    if solution_size > 0:
         solution_network_on_original_A = A_network[np.ix_(est_subnetwork_indices, est_subnetwork_indices)]
         num_edges_in_solution_on_original_A = sum(sum(solution_network_on_original_A))/2
         # Avoid division by zero if solution_size is 0 or 1
         if solution_size > 1:
              solution_network_density_on_original_A = num_edges_in_solution_on_original_A / (solution_size*(solution_size-1)/2)


    print("\n--- NetMix Results ---")
    print("Number of vertices in identified subnetwork: {}".format(solution_size))
    # Report density on the original processed network
    print("Density of identified subnetwork (on original network): {:.4f}".format(solution_network_density_on_original_A))
    # Optionally report density on G_delta
    # solution_network_on_delta = sim_mat_delta[np.ix_(est_subnetwork_indices, est_subnetwork_indices)] if solution_size > 0 else None
    # num_edges_in_solution_on_delta = sum(sum(solution_network_on_delta))/2 if solution_size > 0 else 0
    # solution_network_density_on_delta = num_edges_in_solution_on_delta / (solution_size*(solution_size-1)/2) if solution_size > 1 else 0
    # print("Density of identified subnetwork (on G_delta): {:.4f}".format(solution_network_density_on_delta))


    ###########################################################
    # 13. write solution
    if args.output:
        try:
            os.makedirs(args.output, exist_ok=True)
            output_subnetwork_file = os.path.join(args.output, 'netmix_subnetwork.tsv')
            output_nodelist_file = os.path.join(args.output, 'node_list.tsv') # This will be the processed node list
            write_list_to_file(output_subnetwork_file, sorted(est_subnetwork_genes)) # Write sorted gene names
            write_list_to_file(output_nodelist_file, node_list) # Write the processed node list
            print(f"Identified subnetwork genes written to: {output_subnetwork_file}")
            print(f"Processed node list written to: {output_nodelist_file}")
        except Exception as e:
            print(f"Error writing output files to {args.output}: {e}")
    else:
        if verbosity > 0:
            print("No output directory specified. Results not written to file.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args() # parse_args() by default uses sys.argv[1:]
    run(args)

