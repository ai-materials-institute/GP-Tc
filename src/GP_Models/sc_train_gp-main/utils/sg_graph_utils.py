#Version directly using space group number to label nodes, which is consistant with the ML data

import pandas as pd
import numpy as np 
import networkx as nx
import re

def parse_subgroup_entry(entry):
    """
    Parse a maximal subgroup entry from the Excel file.

    Args:
        entry (str): A string like '2(3)', representing space group number 2 with index 3.

    Returns:
        Tuple[int, int]: Subgroup number and index.
    """
    match = re.match(r'(\d+)\((\d+)\)', str(entry))
    if match:
        subgroup_number = int(match.group(1))
        index = int(match.group(2))
        return subgroup_number, index
    return None, None

def read_maximal_subgroups(filename):
    """
    Read maximal subgroup relationships from an Excel file.

    Args:
        filename (str): Path to the Excel file.

    Returns:
        Tuple[Dict[int, List[Tuple[int, int]]], List[int]]: 
        - maximal_subgroups_dict: Mapping from space group number to its maximal subgroups and indices.
        - sg_list_in_order: List of space group numbers in order.
    """
    df = pd.read_excel(filename, header=None, skiprows=1)  # Skip the first row (header)
    maximal_subgroups_dict = {}
    sg_list_in_order = []

    for idx, row in df.iterrows():
        sg_number = row[0]  # Space group number
        if pd.isna(sg_number):
            continue

        sg_number = int(sg_number)  # Ensure it's an integer
        sg_list_in_order.append(sg_number)
        maximal_subgroups = []

        for entry in row[1:]:
            if pd.isna(entry):
                continue
            subgroup_number, index = parse_subgroup_entry(entry)
            if subgroup_number:
                maximal_subgroups.append((subgroup_number, index))

        maximal_subgroups_dict[sg_number] = maximal_subgroups

    return maximal_subgroups_dict, sg_list_in_order

def create_space_group_graph(filename):
    """
    Generate a directed graph representing the maximal subgroup relationships of space groups.

    Args:
        filename (str): Path to the Excel file.

    Returns:
        Tuple[networkx.DiGraph, List[int], Dict[int, List[Tuple[int, int]]]]]: 
        - Directed graph G
        - List of space group numbers in order
        - Dictionary mapping space group numbers to their maximal subgroups
    """
    maximal_subgroups_dict, sg_list_in_order = read_maximal_subgroups(filename)
    G = nx.DiGraph()

    # Add all space groups as nodes
    for sg_number in sg_list_in_order:
        G.add_node(sg_number)

    # Add edges (maximal subgroup relationships)
    for sg_number, subgroups in maximal_subgroups_dict.items():
        for subgroup_number, index in subgroups:
            if subgroup_number not in G:
                G.add_node(subgroup_number)
            G.add_edge(sg_number, subgroup_number, weight=index)

    return G, sg_list_in_order, maximal_subgroups_dict

def extract_mutual_edges(G):
    """
    Extract pairs of nodes that have mutual edges in the directed graph.

    Args:
        G (networkx.DiGraph): The directed graph.

    Returns:
        List[Tuple[int, int, int, int]]: List of node pairs with mutual edges and their indices.
    """
    mutual_edges = []
    for u, v in G.edges():
        if G.has_edge(v, u):
            weight_uv = G[u][v]['weight']
            weight_vu = G[v][u]['weight']
            if u <= v:  # Store only once
                mutual_edges.append((u, v, weight_uv, weight_vu))
    return mutual_edges

def extract_mutual_subgroups_from_excel(maximal_subgroups_dict):
    """
    Extract pairs of space groups that are mutual maximal subgroups from the Excel data.

    Args:
        maximal_subgroups_dict (dict): Dictionary mapping space groups to their maximal subgroups.

    Returns:
        Set[Tuple[int, int]]: Set of pairs of mutual maximal subgroups.
    """
    mutual_subgroups = set()
    for sg, subgroups in maximal_subgroups_dict.items():
        for subgroup, _ in subgroups:
            if sg in [s for s, _ in maximal_subgroups_dict.get(subgroup, [])]:
                pair = tuple(sorted([sg, subgroup]))
                mutual_subgroups.add(pair)
    return mutual_subgroups

def compare_graph_and_excel_mutuals(G, maximal_subgroups_dict):
    """
    Compare the mutual maximal subgroup relationships found in the graph with those in the Excel file.

    Args:
        G (networkx.DiGraph): The directed graph.
        maximal_subgroups_dict (dict): Dictionary mapping space groups to their maximal subgroups.

    Returns:
        None
    """
    mutual_edges = extract_mutual_edges(G)
    graph_mutual_pairs = set(tuple(sorted([u, v])) for u, v, _, _ in mutual_edges)
    excel_mutual_pairs = extract_mutual_subgroups_from_excel(maximal_subgroups_dict)

    only_in_graph = graph_mutual_pairs - excel_mutual_pairs
    only_in_excel = excel_mutual_pairs - graph_mutual_pairs
    in_both = graph_mutual_pairs & excel_mutual_pairs

    print("Mutual maximal subgroup pairs found in both the graph and Excel file:")
    for pair in sorted(in_both):
        print(f"  {pair}")

    if only_in_graph:
        print("\nMutual maximal subgroup pairs found only in the graph:")
        for pair in sorted(only_in_graph):
            print(f"  {pair}")

    if only_in_excel:
        print("\nMutual maximal subgroup pairs found only in the Excel file:")
        for pair in sorted(only_in_excel):
            print(f"  {pair}")

    if not only_in_graph and not only_in_excel:
        print("\nThe graph and Excel file data are completely consistent!")

def check_edge_weights(G, mutual_edges):
    """
    Check the weights of mutual edges in the graph.

    Args:
        G (networkx.DiGraph): The directed graph.
        mutual_edges (List[Tuple[int, int, int, int]]): List of mutual edges with weights.

    Returns:
        None
    """
    print("\nMutual edges and their indices:")
    for u, v, weight_uv, weight_vu in mutual_edges:
        print(f"  {u} → {v} (index {weight_uv})")
        print(f"  {v} → {u} (index {weight_vu})\n")


MAX_SHORTEST_PATH_DIST = 5 # max dist apart of any pair of graph nodes 
def construct_distance_matrix(G):
    n_nodes = len(G.nodes())
    value_for_unconnected_nodes = MAX_SHORTEST_PATH_DIST + 1 
    dist_matrix = np.ones((n_nodes, n_nodes))*10_000_000 # (230, 230)
    for node1 in G.nodes():
        for node2 in G.nodes():
            try:
                dist = nx.shortest_path_length(G, source=node1, target=node2)
                # dist2 = nx.shortest_path_length(G, source=node2, target=node1)
                # dist = min(dist1, dist2)
            except nx.exception.NetworkXNoPath:
                dist = value_for_unconnected_nodes
            curr_dist = dist_matrix[node1 - 1, node2 - 1]
            dist = min(curr_dist, dist)
            dist_matrix[node1 - 1, node2 - 1] = dist 
            dist_matrix[node2 - 1, node1 - 1] = dist 
    assert dist_matrix.min() == 0
    assert dist_matrix.max() <= value_for_unconnected_nodes

    return dist_matrix 


def get_sg_graph_distance_matrix():
    filename = "../data_w_sg/maximal_subgroup_nonsym.xlsx"
    G, _, _ = create_space_group_graph(filename)
    dist_matrix = construct_distance_matrix(G)
    return dist_matrix


if __name__ == "__main__":
    # Load and create graph 
    filename = "../data_w_sg/maximal_subgroup_nonsym.xlsx"
    G, sg_list_in_order, maximal_subgroups_dict = create_space_group_graph(filename)

    # Count edges
    edge_count = G.number_of_edges()
    print(f"Total number of edges: {edge_count}")

    # Extract mutual maximal subgroup edges
    mutual_edges = extract_mutual_edges(G)

    # Count mutual edges
    mutual_edges_count = len(mutual_edges)
    print(f"Number of mutual edges: {mutual_edges_count}")

    # Check edge weights
    check_edge_weights(G, mutual_edges)

    # Compare graph with Excel data
    compare_graph_and_excel_mutuals(G, maximal_subgroups_dict)
    
    dist_matrix = construct_distance_matrix(G)
    print("Distance Matrix:", dist_matrix.shape, dist_matrix.min(), dist_matrix.max())
