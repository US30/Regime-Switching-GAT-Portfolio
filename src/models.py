import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATPortfolioAgent(nn.Module):
    def __init__(self, num_features, hidden_channels=64, num_heads=4, dropout=0.2):
        """
        A Graph Attention Network that outputs portfolio weights.
        
        Args:
            num_features (int): Number of input features per stock (e.g., Return, Volatility).
            hidden_channels (int): Size of the hidden layers.
            num_heads (int): Number of attention heads (allows model to focus on different things).
            dropout (float): Regularization to prevent overfitting.
        """
        super().__init__()
        
        # Layer 1: GATv2 is more expressive than standard GAT
        # We use edge_dim=1 because our edges have weights (Correlations)
        self.gat1 = GATv2Conv(
            in_channels=num_features, 
            out_channels=hidden_channels, 
            heads=num_heads, 
            dropout=dropout, 
            edge_dim=1 
        )
        
        # Layer 2: Aggregating the heads from Layer 1
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * num_heads, 
            out_channels=hidden_channels, 
            heads=1, 
            dropout=dropout, 
            edge_dim=1
        )
        
        # Final Linear Layer: Projects hidden state to a single "score" per stock
        self.lin = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the neural network.
        
        Args:
            x: Node features [Num_Nodes, Num_Features]
            edge_index: Graph connectivity [2, Num_Edges]
            edge_attr: Edge weights (correlations) [Num_Edges, 1]
        """
        # 1. First Graph Attention Layer
        # The model "looks" at neighbors and aggregates information based on correlation
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) # ELU activation is common in GATs
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 2. Second Graph Attention Layer
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # 3. Scoring Layer
        # Output shape: [Num_Nodes, 1] -> Raw score for each stock
        scores = self.lin(x)
        
        # 4. Portfolio Allocation (Softmax)
        # Softmax ensures all weights sum to 1.0 (Long-Only constraint)
        # If you wanted Short selling, you would use Tanh instead.
        weights = F.softmax(scores, dim=0)
        
        return weights