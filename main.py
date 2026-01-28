import matplotlib.pyplot as plt
import yaml
from src.data_loader import fetch_and_process
from src.regime import get_regime_states

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def plot_regimes(df, ticker):
    """
    Visualizes the market regimes.
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['green', 'orange', 'red']
    labels = ['Low Vol (Bull)', 'Medium Vol', 'High Vol (Bear)']
    
    for regime in range(3):
        mask = df['regime'] == regime
        plt.scatter(df.index[mask], df['price'][mask], 
                   s=10, c=colors[regime], label=labels[regime], alpha=0.6)
    
    plt.title(f"{ticker} Regime Detection (Green=Safe, Red=Crash)", fontsize=14)
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Fetch Data
    ticker = config['data']['ticker']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    df = fetch_and_process(ticker, start, end)
    
    # 2. Detect Regimes
    df_regime = get_regime_states(df, n_components=config['regime_model']['n_components'])
    
    # 3. Print Stats
    print("\n--- Regime Statistics ---")
    print(df_regime.groupby('regime')[['log_ret', 'volatility']].mean())
    
    # 4. Plot
    plot_regimes(df_regime, ticker)