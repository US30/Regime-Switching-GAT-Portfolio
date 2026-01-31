import torch
import torch.optim as optim
from src.loss import MeanVarianceLoss

def train_model(model, data_obj, returns_tensor, epochs=500, lr=0.01):
    """
    Trains the GAT model to maximize Sharpe Ratio on the provided data.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = SharpeLoss()
    
    # Store history for plotting
    loss_history = []
    
    model.train() # Set model to training mode
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad() # Clear old gradients
        
        # 1. Forward Pass: Get Weights from GNN
        # Reshape edge_attr for GAT [Edges, 1]
        edge_attr = data_obj.edge_attr.view(-1, 1)
        weights = model(data_obj.x, data_obj.edge_index, edge_attr)
        
        # Flatten weights to [Num_Assets]
        weights = weights.flatten()
        
        # 2. Compute Loss (Negative Sharpe)
        # We evaluate how well these weights would have performed on the returns
        loss = loss_fn(weights, returns_tensor)
        
        # 3. Backward Pass (Learn)
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss (Neg Sharpe): {loss.item():.4f} | Best Sharpe: {-loss.item():.4f}")
            
    return model, loss_history