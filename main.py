import matplotlib.pyplot as plt
import yaml
import torch
from src.data_loader import fetch_and_process, fetch_portfolio_data
from src.regime import get_regime_states
from src.graph import build_graph_from_correlations, plot_graph

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    # --- Step 1: Detect Global Regimes (using SPY) ---
    print("\n--- Step 1: Market Regime Detection ---")
    spy_df = fetch_and_process(config['data']['ticker'], config['data']['start_date'], config['data']['end_date'])
    spy_df, _ = get_regime_states(spy_df)
    
    # Identify the "Crash" periods (Regime 2)
    crash_dates = spy_df[spy_df['regime'] == 2].index
    print(f"Identified {len(crash_dates)} days classified as 'High Volatility/Crash'.")

    # --- Step 2: Build Portfolio Graph ---
    print("\n--- Step 2: Building Asset Graph ---")
    # Let's define a small portfolio of Tech + Defense + Retail
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'JPM', 'XOM', 'GLD']
    
    portfolio_returns = fetch_portfolio_data(tickers, config['data']['start_date'], config['data']['end_date'])
    
    # Filter portfolio data to ONLY look at the "Crash" days
    # We intersect the portfolio dates with the crash dates
    crash_returns = portfolio_returns.loc[portfolio_returns.index.intersection(crash_dates)]
    
    print(f"Building Graph based on {len(crash_returns)} crash days...")
    
    # Build Graph with a high correlation threshold
    # In a crash, correlations usually spike.
    data_obj, G = build_graph_from_correlations(crash_returns, threshold=0.6)
    
    print(f"Graph Created: {data_obj.num_nodes} Nodes, {data_obj.num_edges} Edges.")
    
    # --- Step 3: Visualize ---
    plot_graph(G, title="Asset Correlations during 'Crash' Regimes (Regime 2)")