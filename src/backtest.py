import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(daily_returns):
    """
    Computes key financial metrics.
    """
    # Cumulative Return
    cumulative = (1 + daily_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    # Annualized Sharpe (assuming 252 trading days)
    mean_ret = daily_returns.mean() * 252
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = mean_ret / (volatility + 1e-6)
    
    # Max Drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return {
        "Total Return": f"{total_return:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}"
    }

def plot_performance(strategy_rets, benchmark_rets, title="Strategy vs Benchmark"):
    """
    Plots the equity curve.
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate Equity Curves (starting at $1.00)
    strat_curve = (1 + strategy_rets).cumprod()
    bench_curve = (1 + benchmark_rets).cumprod()
    
    plt.plot(strat_curve, label="GNN Regime Strategy", color='blue', linewidth=2)
    plt.plot(bench_curve, label="Benchmark (Equal Weight)", color='gray', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()