import torch

class SharpeLoss(torch.nn.Module):
    def __init__(self, risk_free_rate=0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, weights, asset_returns):
        """
        Computes the Negative Sharpe Ratio.
        
        Args:
            weights: [Num_Assets] vector (output of GNN) from Softmax.
            asset_returns: [Time_Steps, Num_Assets] matrix of future returns.
        """
        # 1. Calculate Portfolio Returns over the period
        # Rp = weights * Asset_Returns
        # Shape: [Time_Steps]
        portfolio_ret = torch.matmul(asset_returns, weights)
        
        # 2. Calculate Expected Return (Mean) and Risk (Std Dev)
        expected_ret = torch.mean(portfolio_ret)
        volatility = torch.std(portfolio_ret)
        
        # 3. Compute Sharpe Ratio
        # Add epsilon 1e-6 to volatility to avoid division by zero
        sharpe_ratio = (expected_ret - self.risk_free_rate) / (volatility + 1e-6)
        
        # 4. We want to MAXIMIZE Sharpe, so we MINIMIZE Negative Sharpe
        return -sharpe_ratio