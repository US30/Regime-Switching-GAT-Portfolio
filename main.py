import matplotlib.pyplot as plt
import yaml
import torch
import pandas as pd
import numpy as np
from src.data_loader import fetch_and_process, fetch_portfolio_data
from src.regime import get_regime_states
from src.graph import build_graph_from_correlations
from src.models import GATPortfolioAgent
from src.train import train_model
from src.backtest import calculate_metrics, plot_performance # <--- NEW IMPORT

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_weights(model, data_obj):
    """Helper to extract weights from the GNN"""
    model.eval()
    with torch.no_grad():
        edge_attr = data_obj.edge_attr.view(-1, 1)
        weights = model(data_obj.x, data_obj.edge_index, edge_attr)
    return weights.flatten().numpy()

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. PREPARE DATA
    # ------------------------------------------------------------------
    print("\n--- Step 1: Data Prep & Regime Detection ---")
    
    # A. Fetch Market Data (SPY) for Regimes
    spy_df = fetch_and_process(config['data']['market_ticker'], config['data']['start_date'], config['data']['end_date'])
    spy_df = get_regime_states(spy_df)
    
    # B. Fetch Portfolio Data
    tickers = config['data']['portfolio_tickers']
    portfolio_rets = fetch_portfolio_data(tickers, config['data']['start_date'], config['data']['end_date'])
    
    # Align dates (Intersection of SPY and Portfolio)
    common_dates = spy_df.index.intersection(portfolio_rets.index)
    spy_df = spy_df.loc[common_dates]
    portfolio_rets = portfolio_rets.loc[common_dates]
    
    # C. Split Train (2015-2020) vs Test (2021-2023)
    train_mask = portfolio_rets.index < '2021-01-01'
    test_mask = portfolio_rets.index >= '2021-01-01'
    
    print(f"Train Days: {sum(train_mask)} | Test Days: {sum(test_mask)}")

    # ------------------------------------------------------------------
    # 2. TRAIN TWO MODELS (Regime Specific)
    # ------------------------------------------------------------------
    print("\n--- Step 2: Training Regime-Specific Models ---")
    
    # --- MODEL A: BULL MARKET (Regime 0) ---
    print("Training Bull Model (Regime 0)...")
    bull_dates = spy_df[(spy_df['regime'] == 0) & train_mask].index
    bull_data = portfolio_rets.loc[bull_dates]
    
    # Build Graph & Train
    data_bull, _ = build_graph_from_correlations(bull_data, threshold=0.3)
    model_bull = GATPortfolioAgent(data_bull.x.shape[1], hidden_channels=32)
    model_bull, _ = train_model(model_bull, data_bull, torch.tensor(bull_data.values, dtype=torch.float), epochs=100)
    w_bull = get_weights(model_bull, data_bull)

    # --- MODEL B: BEAR MARKET (Regime 2) ---
    print("Training Bear Model (Regime 2)...")
    bear_dates = spy_df[(spy_df['regime'] == 2) & train_mask].index
    bear_data = portfolio_rets.loc[bear_dates]
    
    # Build Graph & Train
    data_bear, _ = build_graph_from_correlations(bear_data, threshold=0.5) # Higher threshold in crash
    model_bear = GATPortfolioAgent(data_bear.x.shape[1], hidden_channels=32)
    model_bear, _ = train_model(model_bear, data_bear, torch.tensor(bear_data.values, dtype=torch.float), epochs=100)
    w_bear = get_weights(model_bear, data_bear)

    # Print Learned Allocations
    print("\n[Allocations Learned]")
    alloc_df = pd.DataFrame({'Ticker': tickers, 'Bull_W': w_bull, 'Bear_W': w_bear})
    alloc_df['Bull_W'] = alloc_df['Bull_W'].apply(lambda x: f"{x:.1%}")
    alloc_df['Bear_W'] = alloc_df['Bear_W'].apply(lambda x: f"{x:.1%}")
    print(alloc_df)

    # ------------------------------------------------------------------
    # 3. BACKTEST (WALK-FORWARD)
    # ------------------------------------------------------------------
    print("\n--- Step 3: Running Backtest (2021-2023) ---")
    
    test_rets = portfolio_rets.loc[test_mask]
    test_regimes = spy_df.loc[test_mask, 'regime']
    
    # Strategy Logic:
    # If Regime 2 (Crash) -> Use Bear Weights
    # Else -> Use Bull Weights
    
    strategy_returns = []
    
    for date, daily_ret in test_rets.iterrows():
        today_regime = test_regimes.loc[date]
        
        if today_regime == 2: # Crash
            weights = w_bear
        else:
            weights = w_bull
            
        # Daily Portfolio Return = sum(weight * asset_return)
        day_p_ret = np.dot(weights, daily_ret.values)
        strategy_returns.append(day_p_ret)
        
    # Create Series
    strat_series = pd.Series(strategy_returns, index=test_rets.index)
    
    # Benchmark: Equal Weight Portfolio (1/N)
    bench_series = test_rets.mean(axis=1)

    # ------------------------------------------------------------------
    # 4. RESULTS
    # ------------------------------------------------------------------
    print("\n--- Performance Metrics ---")
    print("Strategy:", calculate_metrics(strat_series))
    print("Benchmark:", calculate_metrics(bench_series))
    
    plot_performance(strat_series, bench_series)