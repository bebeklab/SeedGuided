import numpy as np
from gurobipy import *

# --- seedmix Optimization Functions ---

def seedmix_edgedense(A, rho, resps, seed_nodes=None, alpha=0.05, edge_dense_quad=False, edge_dense_linear=False, output=False, time_limit=None, method=None, thread_count = None, mipgap =None):
    """
    Finds a dense subnetwork in a graph by solving a Gurobi MIP,
    including a specified set of seed nodes.

    Args:
        A (np.ndarray): Adjacency matrix of the graph (typically G_delta from PPR).
        rho (float): Target edge density value for the constraint.
        resps (np.ndarray or list or dict): Node response values to maximize (e.g., z-scores).
        seed_nodes (list or set, optional): List of integer indices of nodes
            to force into the subnetwork. Indices must correspond to rows/cols in A.
            Defaults to None (no forced seeds).
        alpha (float): Parameter to determine the maximum clique size (alpha * n).
        edge_dense_quad (bool): Use quadratic edge density constraint (default).
        edge_dense_linear (bool): Use linear edge density constraint (requires Gurobi 9+).
                                   Note: Linear formulation here might not be standard density.
        output (bool): Whether to show Gurobi solver output.
        time_limit (float, optional): Time limit (in seconds) for Gurobi solver.
        method (int, optional): Gurobi Method parameter.
        thread_count (int, optional): Gurobi Threads parameter.
        mipgap (float, optional): Gurobi MIPGap parameter.

    Returns:
        list: Indices of the nodes in the estimated subnetwork, or an empty list
              if optimization fails or finds no nodes.
    """
    n = A.shape[0]
    if n == 0:
        print("Error: Adjacency matrix A is empty in seedmix_edgedense.")
        return []

    # Calculate initial clique size based on alpha and network size
    initial_clique_size = int(n*alpha)

    # Ensure seed nodes are handled; if None, treat as empty list
    seed_nodes = set(seed_nodes) if seed_nodes is not None else set()

    # Validate seed node indices against the current network size
    valid_seed_nodes = {i for i in seed_nodes if 0 <= i < n}
    invalid_seed_nodes = seed_nodes - valid_seed_nodes
    if invalid_seed_nodes:
        print(f"Warning: Skipping invalid seed node indices in seedmix_edgedense: {sorted(list(invalid_seed_nodes))}")

    # Adjust clique size to be at least the number of valid seed nodes
    # This ensures the size constraint doesn't make the model infeasible if seeds > initial_clique_size
    clique_size = max(initial_clique_size, len(valid_seed_nodes))

    print('seedmix_edgedense: n: {}, requested initial_clique_size: {}, effective_clique_size: {}, num_seed_nodes: {}'.format(n, initial_clique_size, clique_size, len(valid_seed_nodes)))

    # set up gurobi
    vertex_inds = list(range(n)) # Use list(range(n)) for consistency

    # Find upper triangle edge indices (assuming symmetric A and no self-loops)
    # This is used in the density/cut constraints to avoid double counting edges.
    edge_inds_x, edge_inds_y = np.nonzero(A)
    edge_inds = [(edge_inds_x[t], edge_inds_y[t]) for t in range(len(edge_inds_x)) if edge_inds_x[t] > edge_inds_y[t]]


    # Create a new model
    m = Model("anomaly_edgedense")
    if not output:
        m.setParam('OutputFlag', 0)
    if time_limit is not None:
        m.setParam('TimeLimit', time_limit) # Time limit is in seconds
    if method is not None:
        m.setParam('Method', method)
    if thread_count is not None:
        m.setParam("Threads", thread_count)
    if mipgap is not None:
        m.setParam("MIPGap", mipgap)

    # Create binary variables for node selection
    x=m.addVars(vertex_inds, vtype=GRB.BINARY, name="x")

    # Add constraints to force seed nodes to be included
    if valid_seed_nodes:
        # print(f"seedmix_edgedense: Forcing inclusion of seed nodes (indices): {sorted(list(valid_seed_nodes))}")
        for seed_node_idx in valid_seed_nodes:
             m.addConstr(x[seed_node_idx] == 1, name=f"force_seed_{seed_node_idx}")

    # create objective: Maximize the sum of responses for selected nodes
    # Ensure resps is a dictionary or can be accessed by index
    if isinstance(resps, np.ndarray) or isinstance(resps, list):
         w = {i: resps[i] for i in vertex_inds}
    elif isinstance(resps, dict):
         w = resps
    else:
         print("Error: Invalid format for resps. Must be list, numpy array, or dictionary.")
         return []

    obj_exp = x.prod(w)

    m.setObjective(obj_exp, GRB.MAXIMIZE)

    # size constraint: Limit the total number of selected nodes
    m.addConstr( quicksum([x[i] for i in vertex_inds]) <= clique_size, name="size_limit" )

    # edge density constraint
    # The constraint is: Sum(A[i,j] for selected i,j) >= rho * PossibleEdges(selected_size)
    # PossibleEdges(k) = k*(k-1)/2
    # Sum(A[i,j] for selected i,j) is sum(A[i,j]*x[i]*x[j] for i,j in edges)
    vecsum = quicksum( [x[i] for i in vertex_inds] ) # Sum of selected nodes (size of the set)
    if edge_dense_quad:
        # Quadratic form: sum(A[i,j]*x[i]*x[j]) for i > j
        LHS = quicksum( [ A[i,j]*x[i]*x[j] for (i,j) in edge_inds ] )
        # RHS: rho * k*(k-1)/2 where k = vecsum
        RHS = rho * 0.5*( vecsum*vecsum - vecsum )
        m.addConstr( LHS >= RHS, name="edge_density_quad" )
    elif edge_dense_linear:
         # Linear form as provided in the original code. Note this is likely NOT standard density.
         LHS = quicksum( [ A[i,j]*x[i]*x[j] for (i,j) in edge_inds ] )
         RHS = rho * 0.5 * ( vecsum )
         m.addConstr( LHS >= RHS , name="edge_density_linear")
    else:
        # If neither quadratic nor linear is specified, the density constraint is effectively skipped.
        # This might be intended if you only want size and seed constraints with objective maximization.
        if output: # Print a warning if output is enabled
             print("Warning: No edge density constraint specified (edge_dense_quad or edge_dense_linear must be True).")


    # Optimize model
    try:
        m.optimize()
        # Check optimization status
        #OLD        if m.Status == GRB.OPTIMAL or m.Status == GRB.FEASIBLE:
        if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
            # Check if Gurobi actually found at least one solution before timing out
            if m.SolCount > 0:
                estimated_anomaly = [i for i in vertex_inds if x[i].X > 0.5]
                if output:
                     print(f"Optimization successful. Found {len(estimated_anomaly)} nodes.")
                return estimated_anomaly
            else:
                if output:
                     print("Time limit reached, but no feasible solution was found.")
                return []
        else:
            print(f"Optimization ended with status {m.Status} ({GRB.StatusConst(m.Status)}). No optimal or feasible solution found.")
            return []
    except GurobiError as e:
        print(f"Gurobi Error during optimization: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during seedmix_edgedense optimization: {e}")
        return []

def seedmix_cut(A, rho, resps, seed_nodes=None, alpha=0.05, output=True, time_limit=None):
    """
    Finds a subnetwork with a low normalized cut by solving a Gurobi MIP,
    including a specified set of seed nodes.

    Args:
        A (np.ndarray): Adjacency matrix of the graph (typically G_delta from PPR).
        rho (float): Cut parameter for the constraint.
        resps (np.ndarray or list or dict): Node response values to maximize (e.g., z-scores).
        seed_nodes (list or set, optional): List of integer indices of nodes
            to force into the subnetwork. Indices must correspond to rows/cols in A.
            Defaults to None (no forced seeds).
        alpha (float): Parameter to determine the maximum clique size (alpha * n).
        output (bool): Whether to show Gurobi solver output.
        time_limit (float, optional): Time limit (in seconds) for Gurobi solver.

    Returns:
        list: Indices of the nodes in the estimated subnetwork, or an empty list
              if optimization fails or finds no nodes.
    """
    n = A.shape[0]
    if n == 0:
        print("Error: Adjacency matrix A is empty in seedmix_cut.")
        return []

    # Calculate initial clique size based on alpha and network size
    initial_clique_size = int(n*alpha)

    # Ensure seed nodes are handled; if None, treat as empty list
    seed_nodes = set(seed_nodes) if seed_nodes is not None else set()

    # Validate seed node indices against the current network size
    valid_seed_nodes = {i for i in seed_nodes if 0 <= i < n}
    invalid_seed_nodes = seed_nodes - valid_seed_nodes
    if invalid_seed_nodes:
        print(f"Warning: Skipping invalid seed node indices in seedmix_cut: {sorted(list(invalid_seed_nodes))}")

    # Adjust clique size to be at least the number of valid seed nodes
    clique_size = max(initial_clique_size, len(valid_seed_nodes))

    print('seedmix_cut: n: {}, requested initial_clique_size: {}, effective_clique_size: {}, num_seed_nodes: {}'.format(n, initial_clique_size, clique_size, len(valid_seed_nodes)))

    # set up gurobi
    vertex_inds = list(range(n)) # Use list(range(n)) for consistency

    # Find upper triangle edge indices (assuming symmetric A and no self-loops)
    edge_inds_x, edge_inds_y = np.nonzero(A)
    edge_inds = [(edge_inds_x[t], edge_inds_y[t]) for t in range(len(edge_inds_x)) if edge_inds_x[t] > edge_inds_y[t]]

    # Create a new model
    m = Model("anomaly_cut")
    if not output:
        m.setParam('OutputFlag', 0)
    if time_limit is not None:
        m.setParam('TimeLimit', time_limit) # Time limit is in seconds

    # Create binary variables for node selection
    x=m.addVars(vertex_inds, vtype=GRB.BINARY, name="x")

    # Add constraints to force seed nodes to be included
    if valid_seed_nodes:
        # print(f"seedmix_cut: Forcing inclusion of seed nodes (indices): {sorted(list(valid_seed_nodes))}")
        for seed_node_idx in valid_seed_nodes:
            m.addConstr(x[seed_node_idx] == 1, name=f"force_seed_{seed_node_idx}")


    # create objective: Maximize the sum of responses for selected nodes
    # Ensure resps is a dictionary or can be accessed by index
    if isinstance(resps, np.ndarray) or isinstance(resps, list):
         w = {i: resps[i] for i in vertex_inds}
    elif isinstance(resps, dict):
         w = resps
    else:
         print("Error: Invalid format for resps. Must be list, numpy array, or dictionary.")
         return []

    obj_exp = x.prod(w)

    m.setObjective(obj_exp, GRB.MAXIMIZE)

    # size constraint: Limit the total number of selected nodes
    m.addConstr( quicksum([x[i] for i in vertex_inds]) <= clique_size, name="size_limit" )

    # cut constraint: Minimize the number of edges cut, relative to set size
    # The constraint is: Cut(S, V\S) <= rho * |S|
    # Cut(S, V\S) = Sum(A[i,j] for i in S, j not in S)
    # For binary x_i, A[i,j]*(x[i] + x[j] - 2*x[i]*x[j]) is 1 if (i,j) is an edge and one endpoint is in S and the other is not.
    # This sums the edges cut by the partition (S, V\S) where S is the selected set.
    LHS = quicksum([A[i,j]*(x[i] + x[j] - 2*x[i]*x[j]) for (i,j) in edge_inds])
    # The RHS is rho * |S|, where |S| is the size of the selected set (vecsum).
    vecsum = quicksum([x[i] for i in vertex_inds])
    RHS=rho*vecsum
    m.addConstr( LHS <= RHS, name="cut_constraint" )

    # Optimize model
    try:
        m.optimize()
        # Check optimization status
        # OLD One: if m.Status == GRB.OPTIMAL or m.Status == GRB.FEASIBLE:
        if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
            # Check if Gurobi actually found at least one solution before timing out
            if m.SolCount > 0:
                estimated_anomaly = [i for i in vertex_inds if x[i].X > 0.5]
                if output:
                     print(f"Optimization successful. Found {len(estimated_anomaly)} nodes.")
                return estimated_anomaly
            else:
                if output:
                     print("Time limit reached, but no feasible solution was found.")
                return []
        else:
            print(f"Optimization ended with status {m.Status} ({GRB.StatusConst(m.Status)}). No optimal or feasible solution found.")
            return []
    except GurobiError as e:
        print(f"Gurobi Error during optimization: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during seedmix_cut optimization: {e}")
        return []

