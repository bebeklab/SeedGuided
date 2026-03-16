import csv
import pandas as pd
import numpy as np
from scipy.stats import norm
import networkx as nx
import os # Added os for path joining in write_list_to_file

# --- Helper functions for data loading and processing ---

def read_genes_from_file(genes_file):
    """
    Reads a list of gene names from a file (one gene per line, tab-separated).
    Skips header row if it starts with "gene".

    Args:
        genes_file (str): Path to the file containing gene names.

    Returns:
        list: A list of gene names, or None if the file could not be read.
    """
    L = []
    try:
        with open(genes_file, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # Read first row to check for header
            try:
                first_row = next(reader)
                # Check if it looks like a header (case-insensitive check for "gene")
                if not (first_row and first_row[0].lower() == "gene"):
                    if first_row: # Add the first element if it's not a header and not empty
                         L.append(first_row[0])
            except StopIteration:
                pass # File was empty

            for row in reader:
                if row and row[0]: # Ensure row is not empty and first column has content
                    L.append(row[0])
    except FileNotFoundError:
        print(f"Error: File not found at {genes_file}")
        return None
    except Exception as e:
        print(f"Error reading file {genes_file}: {e}")
        return None
    return L


def write_list_to_file(filename, l):
    """
    Writes a list of items to a file, one item per line, tab-separated.

    Args:
        filename (str): Path to the output file.
        l (list): The list of items to write.
    """
    try:
        # Ensure directory exists before writing
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(filename, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for item in l:
                tsv_writer.writerow([str(item)])
    except Exception as e:
        print(f"Error writing list to file {filename}: {e}")


def load_network(edge_list_file, verbosity=0):
    """
    Loads a network from an edge list file and creates an adjacency matrix
    and a list of nodes with their corresponding indices. Assumes undirected graph.

    Args:
        edge_list_file (str): Path to the network edge list file (tab-separated).
        verbosity (int): Verbosity level.

    Returns:
        tuple: (list of node names, numpy adjacency matrix), or (None, None) if loading fails.
    """
    if verbosity > 0:
        print('loading network')
    try:
        # Read the edge list to get all unique nodes
        df_el = pd.read_csv(edge_list_file, header = None, sep = "\t")
        # Handle potential empty file case
        if df_el.empty:
            if verbosity > 0:
                 print(f"Warning: Edge list file {edge_list_file} is empty.")
            return [], np.zeros((0,0))

        # Get unique nodes from both columns and sort them for consistent indexing
        node_list = sorted(list(set(df_el[0]) | set(df_el[1])))
        node_to_index = {node: i for i, node in enumerate(node_list)}
        num_nodes = len(node_list)

        if num_nodes == 0:
             if verbosity > 0:
                  print(f"Warning: No nodes found in edge list file {edge_list_file}.")
             return [], np.zeros((0,0))


        # Initialize adjacency matrix
        A_network=np.zeros((num_nodes, num_nodes))

        # Re-read the file to populate the adjacency matrix using the established indices
        with open(edge_list_file, 'r') as f:
            # Filter out comment lines starting with #
            arrs = [l.rstrip().split("\t") for l in f if not l.startswith("#")]
            for row in arrs:
                if len(row) >= 2: # Ensure row has at least two columns for an edge
                    node1, node2 = row[0], row[1]
                    if node1 in node_to_index and node2 in node_to_index:
                        i = node_to_index[node1]
                        j = node_to_index[node2]
                        A_network[i,j] = 1
                        A_network[j,i] = 1 # Assuming undirected graph
                    else:
                        if verbosity > 1:
                             print(f"Warning: Skipping edge {row} with unknown node(s).")
                else:
                    if verbosity > 1:
                         print(f"Warning: Skipping malformed row in edge list: {row}")


        num_edges = int(np.sum(A_network)/2)

        if verbosity > 0:
            print("Number of nodes: {}, Number of edges: {}".format(num_nodes, num_edges))

        return (node_list, A_network)

    except FileNotFoundError:
        print(f"Error: Edge list file not found at {edge_list_file}")
        return None, None
    except Exception as e:
        print(f"Error loading network from {edge_list_file}: {e}")
        return None, None


def load_pvalues(pvalues_file, node_list, verbosity=0):
    """
    Loads p-values for nodes from a file and aligns them with the provided node list.
    Assumes p-value file is tab-separated with gene name in the first column
    and p-value in the second.

    Args:
        pvalues_file (str): Path to the p-values file.
        node_list (list): The reference list of node names from the network.
        verbosity (int): Verbosity level.

    Returns:
        numpy.ndarray: Array of p-values aligned with node_list, or None if loading fails.
    """
    if verbosity > 0:
        print('loading genescores')

    if not node_list:
        if verbosity > 0:
             print("Warning: Cannot load p-values for an empty node list.")
        return np.array([]) # Return empty array if node list is empty

    node_to_index = {node: i for i, node in enumerate(node_list)}
    pval_list = np.zeros(len(node_list)) # Initialize with zeros (or a default non-significant value)
    found_pvals = set()

    try:
        with open(pvalues_file, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # Try to skip header if present
            try:
                 # Peek at the first row
                 peek = next(reader)
                 # Check if it looks like a header (e.g., first column is "gene" or similar)
                 # This is a heuristic; a more robust check might read a few rows.
                 if peek and peek[0].lower() in ["gene", "id", "name"]:
                     if verbosity > 1:
                          print(f"Skipping potential header row in p-value file: {peek}")
                 else:
                     # If it doesn't look like a header, process this row
                     if len(peek) >= 2:
                         gene = peek[0]
                         try:
                             pval = float(peek[1])
                             if gene in node_to_index:
                                 ind = node_to_index[gene]
                                 pval_list[ind] = pval
                                 found_pvals.add(gene)
                         except ValueError:
                             if verbosity > 1:
                                 print(f"Warning: Skipping row with non-numeric p-value: {peek}")
                     else:
                         if verbosity > 1:
                             print(f"Warning: Skipping malformed row in p-value file: {peek}")

                 # Continue reading the rest of the file
                 for row in reader:
                     if len(row) >= 2:
                         gene = row[0]
                         try:
                             pval = float(row[1])
                             if gene in node_to_index:
                                 ind = node_to_index[gene]
                                 pval_list[ind] = pval
                                 found_pvals.add(gene)
                             # else: # Uncomment for debugging nodes in pval file but not network
                                 # if verbosity > 1:
                                 #     print(f"Warning: P-value for gene '{gene}' found but gene not in network node list.")
                         except ValueError:
                             if verbosity > 1:
                                 print(f"Warning: Skipping row with non-numeric p-value: {row}")
                     else:
                         if verbosity > 1:
                             print(f"Warning: Skipping malformed row in p-value file: {row}")

            except StopIteration:
                 pass # File was empty or only had a header

    except FileNotFoundError:
        print(f"Error: P-values file not found at {pvalues_file}")
        return None
    except Exception as e:
        print(f"Error loading p-values from {pvalues_file}: {e}")
        return None

    if verbosity > 0:
        print(f"Loaded p-values for {len(found_pvals)} genes present in the network.")

    return pval_list


def restrict_to_genes_in_network(pvals_list, node_list, A_network, verbosity=0):
    """
    Restricts the network, node list, and p-values to nodes with p-values > 0
    and then to the largest connected component of the resulting graph.
    Returns the filtered p-values, node list, and adjacency matrix.

    Args:
        pvals_list (numpy.ndarray): Array of p-values aligned with node_list.
        node_list (list): List of node names.
        A_network (numpy.ndarray): Adjacency matrix.
        verbosity (int): Verbosity level.

    Returns:
        tuple: (filtered p-values, filtered node list, filtered adjacency matrix),
               or (np.array([]), np.array([]), np.zeros((0,0))) if no valid nodes remain.
    """
    if pvals_list is None or node_list is None or A_network is None:
        print("Error: Invalid input to restrict_to_genes_in_network.")
        return np.array([]), np.array([]), np.zeros((0,0))

    if len(node_list) == 0:
         if verbosity > 0:
              print("Input node list is empty. Returning empty results.")
         return np.array([]), np.array([]), np.zeros((0,0))

    # find nodes in the network that has scores (pvals > 0)
    # Using a small epsilon because floating point zeros might not be exactly 0
    nodes_with_pval_in_network_indices = np.where(pvals_list > 1e-12)[0] # Use np.where for efficiency

    if len(nodes_with_pval_in_network_indices) == 0:
        if verbosity > 0:
             print("Warning: No nodes with p-values > 0 found in the network.")
        return np.array([]), np.array([]), np.zeros((0,0))


    # Filter based on nodes with p-values
    filtered_node_list = np.array(node_list)[nodes_with_pval_in_network_indices]
    filtered_A_network = A_network[np.ix_(nodes_with_pval_in_network_indices, nodes_with_pval_in_network_indices)]
    filtered_pvals_list = pvals_list[nodes_with_pval_in_network_indices]

    # find largest connected component
    G=nx.Graph(filtered_A_network)
    if not G.nodes():
         if verbosity > 0:
              print("Warning: Graph is empty after filtering for p-values > 0.")
         return np.array([]), np.array([]), np.zeros((0,0))

    # Get the nodes in the largest connected component (LCC)
    # nx.connected_components returns sets of node indices relative to the filtered graph
    connected_components = list(nx.connected_components(G))
    if not connected_components:
         if verbosity > 0:
              print("Warning: No connected components found after filtering.")
         return np.array([]), np.array([]), np.zeros((0,0))

    lcc_nodes_in_filtered_graph = list(max(connected_components, key=len))

    if not lcc_nodes_in_filtered_graph:
        if verbosity > 0:
             print("Warning: LCC is empty after filtering.")
        return np.array([]), np.array([]), np.zeros((0,0))


    # Filter to the LCC
    lcc_node_list = filtered_node_list[lcc_nodes_in_filtered_graph]
    lcc_A_network = filtered_A_network[np.ix_(lcc_nodes_in_filtered_graph, lcc_nodes_in_filtered_graph)]
    lcc_pvals_list = filtered_pvals_list[lcc_nodes_in_filtered_graph]

    n = len(lcc_pvals_list)

    if verbosity > 0:
        print('number of nodes in filtered G (LCC): {}'.format(n))
        print('number of edges in filtered G (LCC): {}'.format(int(np.sum(lcc_A_network)/2)))

    return (lcc_pvals_list, lcc_node_list, lcc_A_network)


def compute_zscores(pvalues):
    """
    Computes z-scores from p-values using the inverse normal CDF.
    Handles p-values of 0 or 1 by clamping to a small epsilon.

    Args:
        pvalues (numpy.ndarray): Array of p-values.

    Returns:
        numpy.ndarray: Array of z-scores.
    """
    # Handle p-values that are exactly 0 or 1 to avoid infinite z-scores
    # Replace 0 with a very small number, replace 1 with a number very close to 1
    pvalues = np.maximum(pvalues, 1e-16) # Replace 0 with 1e-16
    pvalues = np.minimum(pvalues, 1 - 1e-16) # Replace 1 with 1 - 1e-16

    # compute z-scores, using -1* for significance in the positive direction
    # norm.ppf(p) gives the z-score such that P(Z <= z) = p.
    # For small p-values (high significance), we want high positive z-scores.
    # So we use -1 * norm.ppf(p).
    zscores = -1 * norm.ppf(pvalues)

    return zscores

def post_process_zscores(zscores, verbosity=0):
    """
    Corrects for -np.inf values in z-scores (from p-values of 1)
    by replacing them with the minimum non-infinite z-score.

    Args:
        zscores (numpy.ndarray): Array of z-scores.
        verbosity (int): Verbosity level.

    Returns:
        numpy.ndarray: Array of post-processed z-scores.
    """
    # Find non-infinite z-scores
    finite_zscores = zscores[np.isfinite(zscores)]

    if len(finite_zscores) == 0:
        if verbosity > 0:
             print("Warning: All z-scores are infinite or NaN. Cannot post-process.")
        # Return the original zscores if all are infinite or NaN
        # Or return zeros/a default value depending on how you want to handle this edge case.
        return zscores

    min_finite_zscore = np.min(finite_zscores)
    if verbosity > 0:
        print(f"Minimum finite z-score: {min_finite_zscore}")

    # Replace -inf with the minimum finite z-score
    zscores[zscores == -np.inf] = min_finite_zscore
    # Also replace NaN if any slipped through, maybe with min_finite_zscore or 0
    zscores[np.isnan(zscores)] = min_finite_zscore # Or 0, depending on desired behavior

    return zscores

# The correct_nans_from_locfdr function is specific to locfdr output.
# Include it here if you use locfdr elsewhere or might use its output as responses.
# If you are only using post-processed z-scores as responses, this function
# is not strictly needed for the main script's core logic.
# Keeping it for completeness as it was in your original code.
def correct_nans_from_locfdr(r_locfdr, scores, nulltype_name="mlest", verbosity=0):
    """
    Corrects NaN values and potentially misclassified non-nulls in locfdr results.
    Returns an estimate of the number of non-nulls.
    Note: This function modifies the 'fdr' array within r_locfdr['fdr'].

    Args:
        r_locfdr (dict): Result dictionary from locfdr.locfdr.
        scores (numpy.ndarray): Array of scores (e.g., z-scores) used for locfdr.
        nulltype_name (str): Name of the null type used in locfdr ('mlest', 'uest', etc.).
        verbosity (int): Verbosity level.

    Returns:
        int: Corrected estimate of the number of non-nulls.
    """
    if r_locfdr is None or 'fdr' not in r_locfdr or 'fp0' not in r_locfdr or nulltype_name not in r_locfdr['fp0']['delta']:
         print("Error: Invalid input to correct_nans_from_locfdr.")
         return 0

    resps = 1 - r_locfdr['fdr'] # This is a view, modifications affect r_locfdr['fdr']
    mu = r_locfdr['fp0']["delta"][nulltype_name]
    nan_count = 0
    nan_count2 = 0
    original_nonnull_count = 0
    nonnull_count = 0

    for ind, t in enumerate(resps):
        if np.isnan(t):
            # If score is high, assume it's non-null (response=1)
            if scores[ind] > mu: # Use mu as a threshold based on null distribution mean
                 resps[ind] = 1
            else: # Otherwise, assume null (response=0)
                 resps[ind] = 0
                 nan_count2 += 1
        # If response is positive but score is below the null mean, assume null
        elif t > 0 and scores[ind] < mu:
            resps[ind] = 0
            if t > 0.5: # Count how many confident non-nulls were flipped to null
                 nan_count += 1

        if resps[ind] > 0.5: # Count non-nulls based on the corrected responses
            nonnull_count += 1
        if not np.isnan(t) and t > 0.5: # Count non-nulls based on original responses (before NaN correction/flipping)
            original_nonnull_count += 1

    # Estimate non-nulls based on locfdr's p0 estimate (optional print)
    # nonnull_count_locfdr_p0 = int((1 - r_locfdr['fp0']["p0"][nulltype_name]) * len(scores))

    if verbosity > 0:
        print(f"Original non-null count (locfdr > 0.5 before correction): {original_nonnull_count}")
        print(f"Corrected non-null count (response > 0.5 after correction): {nonnull_count}")
        print(f"NaNs treated as null: {nan_count2}")
        print(f"Positive responses flipped to null (score < mu): {nan_count}")

    return nonnull_count # Return the corrected non-null count


def compute_ppr_kernel(A_network, verbosity = 0):
    """
    Computes the Personalized PageRank (PPR) kernel and row sums.
    Returns the similarity matrix and row sums.

    Args:
        A_network (numpy.ndarray): Adjacency matrix of the network.
        verbosity (int): Verbosity level.

    Returns:
        tuple: (PPR similarity matrix, PPR row sums), or (None, None) if computation fails.
    """
    if verbosity > 0:
        print('computing the PPR kernel')

    num_nodes = len(A_network)
    if num_nodes == 0:
        print("Warning: Cannot compute PPR kernel for an empty network.")
        return np.zeros((0,0)), np.array([])

    r = 0.4 # Restart probability
    degs_network = np.sum(A_network, axis=1) # Sum degrees along rows

    # Handle nodes with zero degree to avoid division by zero
    # Replace 0 degrees with 1 for the D matrix calculation, but be aware
    # this affects the transition matrix P for isolated nodes.
    # A more robust approach for isolated nodes might be needed depending on PPR definition.
    # For standard PPR, isolated nodes stay at their start node.
    # Setting degree to 1 means P[i,i] = 0 for isolated node i, which isn't ideal.
    # Let's handle isolated nodes explicitly or use a different D matrix if needed.
    # For now, keeping the original approach but noting the potential issue.
    zero_degree_mask = degs_network == 0
    degs_network_safe = np.where(zero_degree_mask, 1, degs_network)

    # Diagonal matrix of inverse degrees
    D_inv = np.diag(1.0 / degs_network_safe)

    # Transition matrix P = D_inv * A
    P_network = D_inv.dot(A_network)

    # For isolated nodes (where original degree was 0), P[i,:] will be all zeros.
    # Standard PPR for isolated nodes means the random walk stays at the node.
    # P[i,i] should be 1 for isolated nodes. Let's correct P.
    P_network[zero_degree_mask, :] = 0 # Set row to zeros first
    P_network[zero_degree_mask, zero_degree_mask] = 1 # Set diagonal to 1 for isolated nodes


    # PPR kernel: (I - (1-r)P)^-1 * r * I
    # Using np.linalg.inv can be numerically unstable for large matrices.
    # For sparse matrices, iterative methods are better.
    # If A_network is large and sparse, consider using scipy.sparse.linalg.spsolve
    # or iterative methods for (I - (1-r)P)x = r*b.
    # For simplicity, sticking to np.linalg.inv for now as in the original code.
    try:
        Identity = np.eye(num_nodes)
        # Ensure the matrix (I - (1-r)P) is well-conditioned
        # Eigenvalues of P are <= 1. (1-r)P has eigenvalues <= 1-r < 1.
        # So (I - (1-r)P) should be invertible.
        PPR_mat = r * np.linalg.inv(Identity - (1-r)*P_network)
    except np.linalg.LinAlgError:
        print("Error: Could not compute inverse for PPR kernel. Matrix might be singular or ill-conditioned.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during PPR kernel computation: {e}")
        return None, None


    # Compute rowsums of PPR mat
    PPR_mat_rowsums = np.sum(PPR_mat, axis=1)

    # Similarity matrix: min(PPR_mat, PPR_mat.T)
    # This makes the similarity matrix symmetric.
    PPR_sim_mat = np.minimum(PPR_mat, PPR_mat.T)

    # return similarity matrix, rowsums
    return PPR_sim_mat, PPR_mat_rowsums