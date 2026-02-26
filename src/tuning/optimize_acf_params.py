"""
Optimize autocorrelation breath detection parameters via grid search

Usage:
    # Quick test (few combinations)
    python src/tuning/optimize_acf_params.py --cache-dir cache/ --quick --jobs 4

    # Full optimization (more combinations)
    python src/tuning/optimize_acf_params.py --cache-dir cache/ --jobs 6
"""

import sys
from pathlib import Path
import argparse
import yaml
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tuning.signal_cache import SignalCache
from src.tuning.optimizer import GridSearchOptimizer


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

DEFAULT_SEARCH_SPACE = {
    # 1. The "Sensitivity" Layer
    'acf_min_prominence': [0.03, 0.05, 0.12],

    # Smaller windows are often better for fast, jittery birds
    'window_size': [5.0, 8.0, 10.0, 15.0],

    # 3. The "Selection" Logic
    'acf_peak_selection': ['first', 'prominent'],

    # 4. The "Safety" Layer
    'low_correlation_threshold': [0.2, 0.25, 0.3, 0.4]
}

QUICK_SEARCH_SPACE = {
    'acf_min_prominence': [0.10, 0.15, 0.20],  # 3 values
    'min_confidence': [0.3],  # 1 value
}

def main():
    parser = argparse.ArgumentParser(description='Optimize ACF parameters')
    parser.add_argument('--cache-dir', required=True, help='Signal cache directory')
    parser.add_argument('--config', default='configs/default.yaml', help='Base config')
    parser.add_argument('--output', default='configs/tuned_windowed_autocorr.yaml', help='Output config')
    parser.add_argument('--quick', action='store_true', help='Quick search (3 combos)')
    parser.add_argument('--jobs', '-j', type=int, default=6, help='Parallel jobs')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Ensure signal_processing section exists with defaults
    if 'signal_processing' not in config:
        config['signal_processing'] = {
            'fps': 30,
            'bandpass_filter': {'low_freq': 1.5, 'high_freq': 8.0, 'order': 2},
            'breath_counting': {
                'method': 'autocorrelation_windowed',
                'autocorrelation': {}
            }
        }

    # Set method to autocorrelation_windowed
    if 'breath_counting' not in config['signal_processing']:
        config['signal_processing']['breath_counting'] = {}
    config['signal_processing']['breath_counting']['method'] = 'autocorrelation_windowed'

    # Load cached signals
    cache = SignalCache(args.cache_dir)
    signals = cache.get_signals_with_ground_truth()

    print(f"Loaded {len(signals)} cached signals")

    # Select search space
    search_space = QUICK_SEARCH_SPACE if args.quick else DEFAULT_SEARCH_SPACE
    print(f"Search space: {search_space}")

    # Run optimization
    optimizer = GridSearchOptimizer(
        cached_signals=signals,
        search_space=search_space,
        base_config=config
    )

    best_result = optimizer.optimize(verbose=True, n_jobs=args.jobs)

    # Extract results
    best_params = best_result['params']
    best_error = best_result['avg_error']

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")

    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest average error: {best_error*100:.2f}%")

    # Update config with best parameters
    print(f"\n{'='*70}")
    print(f"Updating configuration with best parameters...")

    for param_name, param_value in best_params.items():
        optimizer._set_nested_value(config, param_name, param_value)

    # Add optimization metadata (convert numpy types to Python native types)
    if 'optimization_metadata' not in config:
        config['optimization_metadata'] = {}

    # Convert numpy types to Python native types for YAML serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types"""
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj

    config['optimization_metadata']['autocorrelation'] = convert_numpy_types({
        'method': 'grid_search',
        'search_space': search_space,
        'best_params': best_params,
        'best_error_pct': best_error * 100,
        'num_signals': len(signals),
    })

    # Save tuned config
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Saved optimized configuration to: {output_path}")

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
