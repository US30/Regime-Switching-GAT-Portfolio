import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def get_regime_states(data, n_components=3, iter=1000):
    """
    Fits HMM and returns the DataFrame with a 'regime' column.
    """
    df = data.copy()
    X = df[['log_ret', 'volatility']].values
    
    print(f"Fitting HMM with {n_components} states...")
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=iter, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    df['regime'] = hidden_states
    
    # Sort states: Force 0=Low Vol, 2=High Vol (Crash)
    state_vol_means = df.groupby('regime')['volatility'].mean()
    sorted_states = state_vol_means.sort_values().index
    state_map = {old_label: new_label for new_label, old_label in enumerate(sorted_states)}
    
    df['regime'] = df['regime'].map(state_map)
    
    return df