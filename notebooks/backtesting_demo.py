"""
RadarX Backtesting Demo

Demonstrates walk-forward validation with comprehensive metrics.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from radarx.models.advanced_training import walk_forward_backtest

def main():
    print("=" * 60)
    print("RadarX Backtesting Demo")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 2000
    n_features = 30
    
    print(f"\nGenerating synthetic dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] > 0).astype(int)
    timestamps = np.arange(n_samples)
    
    print(f"  Positive rate: {y.mean():.1%}")
    
    # Run backtest
    print(f"\nRunning walk-forward backtest (5 folds)...")
    results = walk_forward_backtest(
        X=X,
        y=y,
        timestamps=timestamps,
        n_splits=5,
        output_dir=Path('models/backtest_demo')
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\nSamples tested: {results['n_samples']}")
    print(f"Number of folds: {results['n_folds']}")
    
    print("\nPerformance Metrics:")
    metrics = results['overall_metrics']
    
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
    if 'brier_score' in metrics:
        print(f"  Brier Score:       {metrics['brier_score']:.4f}")
    if 'calibration_error' in metrics:
        print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
    if 'log_loss' in metrics:
        print(f"  Log Loss:          {metrics['log_loss']:.4f}")
    
    print("\nPrecision at thresholds:")
    for key, value in metrics.items():
        if key.startswith('precision@'):
            threshold = key.split('@')[1]
            print(f"  P@{threshold}: {value:.4f}")
    
    print("\n✅ Backtest complete!")
    print(f"Results saved to: models/backtest_demo/")
    
    return results

if __name__ == "__main__":
    results = main()
