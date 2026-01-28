import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def build_graph_from_correlations(returns_df, threshold=0.5):
    """
    Constructs a Graph where:
    - Nodes = Stocks
    - Edges = Correlation > threshold
    
    Returns:
    - PyG Data object (edge_index, edge_attr)
    - NetworkX graph (G) for visualization
    """
    # 1. Calculate Correlation Matrix (Pearson)
    corr_matrix = returns_df.corr().abs() # We take absolute value (magnitude matters)
    
    # 2. Convert to PyTorch Geometric Format (COO Format)
    edge_indices = []
    edge_weights = []
    
    num_nodes = len(returns_df.columns)
    stock_names = returns_df.columns.tolist()
    
    # Loop through the matrix to find strong connections
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j: # No self-loops for now
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > threshold:
                    edge_indices.append([i, j])
                    edge_weights.append(corr_val)
    
    # Convert to Tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create PyG Data Object
    # x = Node features (we will add these in Phase 3, for now use placeholder)
    x = torch.eye(num_nodes, dtype=torch.float) 
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Create NetworkX object for plotting
    G = nx.Graph()
    G.add_nodes_from(stock_names)
    for (src, dst), w in zip(edge_indices, edge_weights):
        G.add_edge(stock_names[src], stock_names[dst], weight=w)
        
    return data, G

def plot_graph(G, title="Asset Graph"):
    """
    Visualizes the stock connectivity.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42) # Force-directed layout
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    
    # Draw Edges (Thicker lines = stronger correlation)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], alpha=0.6)
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.show()