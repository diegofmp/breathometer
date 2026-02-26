"""
Parameter optimization utilities

Implements grid search optimization for automated parameter tuning.
Grid search performs exhaustive evaluation over discrete parameter combinations.

Optimization workflow:
1. Load cached signals with ground truth labels
2. Define parameter search space
3. For each parameter combination, evaluate loss
4. Return parameters that minimize loss
"""

import numpy as np
from typing import Dict, List, Any
from itertools import product
import json
from pathlib import Path
from datetime import datetime
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.signal_processing import SignalProcessor


def _evaluate_single_param_combination(
    params: Dict[str, Any],
    cached_signals: List[Dict],
    base_config: Dict,
    param_mapping: Dict[str, List]
) -> Dict[str, Any]:
    """
    Evaluate a single parameter combination (standalone function for multiprocessing)

    This function must be defined at module level to be picklable by multiprocessing.

    Args:
        params: Parameter combination to evaluate
        cached_signals: List of signal data dictionaries
        base_config: Base configuration
        param_mapping: Mapping from parameter names to config paths

    Returns:
        Dictionary with evaluation results
    """
    import copy

    # Create temporary config with these parameters
    temp_config = copy.deepcopy(base_config)

    # Apply parameters to config
    for param_name, param_value in params.items():
        if param_name not in param_mapping:
            raise ValueError(f"Unknown parameter: {param_name}")

        path = param_mapping[param_name]
        current = temp_config
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = param_value

    # Evaluate on each signal
    errors = []
    bpm_results = []

    for signal_data in cached_signals:
        # Create signal processor with these parameters
        processor = SignalProcessor(temp_config['signal_processing'])

        # Process signal
        raw_signal = signal_data['raw_signal']
        fps = signal_data['fps']

        # estimate_breathing_rate returns (bpm, info_dict)
        detected_bpm, info = processor.estimate_breathing_rate(raw_signal, fps)

        ground_truth_bpm = signal_data['ground_truth_bpm']

        # Calculate error
        if ground_truth_bpm > 0:
            error = abs(detected_bpm - ground_truth_bpm) / ground_truth_bpm
        else:
            error = float('inf')

        errors.append(error)
        bpm_results.append({
            'video': Path(signal_data['video_path']).name,
            'detected': detected_bpm,
            'ground_truth': ground_truth_bpm,
            'error_pct': error * 100
        })

    # Calculate aggregate metrics
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)

    # Loss function: mean absolute percentage error + consistency penalty
    consistency_penalty = std_error * 0.1  # Weight std at 10% of loss
    total_loss = avg_error + consistency_penalty

    return {
        'params': params,
        'loss': total_loss,
        'avg_error': avg_error,
        'std_error': std_error,
        'max_error': max_error,
        'bpm_results': bpm_results
    }


class ParameterOptimizer:
    """
    Base class for parameter optimization

    Provides common functionality for evaluating parameters against cached signals.
    """

    def __init__(
        self,
        cached_signals: List[Dict],
        search_space: Dict[str, List[Any]],
        base_config: Dict
    ):
        """
        Initialize optimizer

        Args:
            cached_signals: List of signal data dictionaries with ground truth
            search_space: Dictionary mapping parameter names to value lists
            base_config: Base configuration dictionary (will be updated with search params)
        """
        self.cached_signals = cached_signals
        self.search_space = search_space
        self.base_config = base_config
        self.results = []

        # Validate cached signals have ground truth
        for signal_data in cached_signals:
            if 'ground_truth_bpm' not in signal_data:
                raise ValueError(f"Signal {signal_data['video_path']} missing ground_truth_bpm")

    def evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a parameter set on all cached signals

        Args:
            params: Dictionary of parameter values to test

        Returns:
            Dictionary with evaluation results
        """
        # Create temporary config with these parameters
        temp_config = self._apply_params_to_config(params)

        # Evaluate on each signal
        errors = []
        bpm_results = []

        for signal_data in self.cached_signals:
            # Create signal processor with these parameters
            processor = SignalProcessor(temp_config['signal_processing'])

            # Process signal
            raw_signal = signal_data['raw_signal']
            fps = signal_data['fps']

            # estimate_breathing_rate returns (bpm, info_dict)
            detected_bpm, info = processor.estimate_breathing_rate(raw_signal, fps)

            ground_truth_bpm = signal_data['ground_truth_bpm']

            # Calculate error
            if ground_truth_bpm > 0:
                error = abs(detected_bpm - ground_truth_bpm) / ground_truth_bpm
            else:
                error = float('inf')

            errors.append(error)
            bpm_results.append({
                'video': Path(signal_data['video_path']).name,
                'detected': detected_bpm,
                'ground_truth': ground_truth_bpm,
                'error_pct': error * 100
            })

        # Calculate aggregate metrics
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)

        # Loss function: mean absolute percentage error + consistency penalty
        consistency_penalty = std_error * 0.1  # Weight std at 10% of loss
        total_loss = avg_error + consistency_penalty

        return {
            'params': params,
            'loss': total_loss,
            'avg_error': avg_error,
            'std_error': std_error,
            'max_error': max_error,
            'bpm_results': bpm_results
        }

    def _set_nested_value(self, config: Dict, param_name: str, param_value: Any):
        """
        Set a nested parameter value in a config dictionary

        Args:
            config: Configuration dictionary to modify
            param_name: Parameter name (e.g., 'fft_window_size')
            param_value: Value to set
        """
        # Map parameter names to config paths
        param_mapping = {
            # Bandpass filter parameters
            'bandpass_order': ['signal_processing', 'bandpass_filter', 'order'],
            'bandpass_low_freq': ['signal_processing', 'bandpass_filter', 'low_freq'],
            'bandpass_high_freq': ['signal_processing', 'bandpass_filter', 'high_freq'],
            # Autocorrelation parameters (windowed ACF method)
            'acf_min_prominence': ['signal_processing', 'breath_counting', 'autocorrelation', 'acf_min_prominence'],
            'acf_min_confidence': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_confidence'],
            'min_confidence': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_confidence'],
            'acf_min_bpm': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_breathing_rate_bpm'],
            'acf_max_bpm': ['signal_processing', 'breath_counting', 'autocorrelation', 'max_breathing_rate_bpm'],
            'acf_peak_selection': ['signal_processing', 'breath_counting', 'autocorrelation', 'acf_peak_selection'],
            'low_correlation_threshold': ['signal_processing', 'breath_counting', 'autocorrelation', 'low_correlation_threshold'],
            'acf_window_size': ['signal_processing', 'breath_counting', 'autocorrelation', 'window_size'],
            'window_size': ['signal_processing', 'breath_counting', 'autocorrelation', 'window_size'],
            'acf_overlap': ['signal_processing', 'breath_counting', 'autocorrelation', 'overlap'],
            'overlap': ['signal_processing', 'breath_counting', 'autocorrelation', 'overlap'],
        }

        if param_name not in param_mapping:
            raise ValueError(f"Unknown parameter: {param_name}")

        path = param_mapping[param_name]

        # Navigate to nested location
        current = config
        for key in path[:-1]:
            current = current[key]

        # Set value
        current[path[-1]] = param_value

    def _apply_params_to_config(self, params: Dict[str, Any]) -> Dict:
        """
        Apply parameter values to configuration dictionary

        Args:
            params: Parameter values to apply

        Returns:
            Updated configuration dictionary
        """
        import copy
        config = copy.deepcopy(self.base_config)

        # Map parameter names to config paths
        param_mapping = {
            # Bandpass filter parameters
            'bandpass_order': ['signal_processing', 'bandpass_filter', 'order'],
            'bandpass_low_freq': ['signal_processing', 'bandpass_filter', 'low_freq'],
            'bandpass_high_freq': ['signal_processing', 'bandpass_filter', 'high_freq'],
            # Autocorrelation parameters (windowed ACF method)
            'acf_min_prominence': ['signal_processing', 'breath_counting', 'autocorrelation', 'acf_min_prominence'],
            'acf_min_confidence': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_confidence'],
            'min_confidence': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_confidence'],
            'acf_min_bpm': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_breathing_rate_bpm'],
            'acf_max_bpm': ['signal_processing', 'breath_counting', 'autocorrelation', 'max_breathing_rate_bpm'],
            'acf_peak_selection': ['signal_processing', 'breath_counting', 'autocorrelation', 'acf_peak_selection'],
            'low_correlation_threshold': ['signal_processing', 'breath_counting', 'autocorrelation', 'low_correlation_threshold'],
            'acf_window_size': ['signal_processing', 'breath_counting', 'autocorrelation', 'window_size'],
            'window_size': ['signal_processing', 'breath_counting', 'autocorrelation', 'window_size'],
            'acf_overlap': ['signal_processing', 'breath_counting', 'autocorrelation', 'overlap'],
            'overlap': ['signal_processing', 'breath_counting', 'autocorrelation', 'overlap'],
        }

        # Apply each parameter
        for param_name, param_value in params.items():
            if param_name not in param_mapping:
                raise ValueError(f"Unknown parameter: {param_name}")

            path = param_mapping[param_name]

            # Navigate to nested location
            current = config
            for key in path[:-1]:
                current = current[key]

            # Set value
            current[path[-1]] = param_value

        return config

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization (implemented by subclasses)

        Returns:
            Best parameters and results
        """
        raise NotImplementedError("Subclasses must implement optimize()")

    def save_results(self, output_path: str):
        """
        Save optimization results to JSON file

        Args:
            output_path: Path to output file
        """
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'search_space': self.search_space,
            'num_signals': len(self.cached_signals),
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


class GridSearchOptimizer(ParameterOptimizer):
    """
    Grid search optimizer

    Performs exhaustive search over all parameter combinations.
    Best for small search spaces (<1000 combinations).
    Supports parallel evaluation using multiprocessing.
    """

    def optimize(self, verbose: bool = True, n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run grid search optimization

        Args:
            verbose: Print progress during optimization
            n_jobs: Number of parallel jobs (-1 = use all CPU cores, 1 = sequential)

        Returns:
            Dictionary with best parameters and results
        """
        # Generate all parameter combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        combinations = list(product(*param_values))

        total_combinations = len(combinations)

        # Determine number of workers
        if n_jobs == -1:
            n_workers = cpu_count()
        elif n_jobs == 1:
            n_workers = 1
        else:
            n_workers = min(n_jobs, cpu_count())

        if verbose:
            print(f"\n{'='*70}")
            print(f"GRID SEARCH OPTIMIZATION")
            print(f"{'='*70}")
            print(f"Parameters to tune: {', '.join(param_names)}")
            print(f"Total combinations: {total_combinations}")
            print(f"Signals to evaluate: {len(self.cached_signals)}")
            print(f"Total evaluations: {total_combinations * len(self.cached_signals)}")
            print(f"Parallel workers: {n_workers} (CPU cores: {cpu_count()})")
            print(f"{'='*70}\n")

        # Create parameter mapping (needed for parallel evaluation)
        param_mapping = {
            # Bandpass filter parameters
            'bandpass_order': ['signal_processing', 'bandpass_filter', 'order'],
            'bandpass_low_freq': ['signal_processing', 'bandpass_filter', 'low_freq'],
            'bandpass_high_freq': ['signal_processing', 'bandpass_filter', 'high_freq'],
            # Autocorrelation parameters (windowed ACF method)
            'acf_min_prominence': ['signal_processing', 'breath_counting', 'autocorrelation', 'acf_min_prominence'],
            'acf_min_confidence': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_confidence'],
            'min_confidence': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_confidence'],
            'acf_min_bpm': ['signal_processing', 'breath_counting', 'autocorrelation', 'min_breathing_rate_bpm'],
            'acf_max_bpm': ['signal_processing', 'breath_counting', 'autocorrelation', 'max_breathing_rate_bpm'],
            'acf_peak_selection': ['signal_processing', 'breath_counting', 'autocorrelation', 'acf_peak_selection'],
            'low_correlation_threshold': ['signal_processing', 'breath_counting', 'autocorrelation', 'low_correlation_threshold'],
            'acf_window_size': ['signal_processing', 'breath_counting', 'autocorrelation', 'window_size'],
            'window_size': ['signal_processing', 'breath_counting', 'autocorrelation', 'window_size'],
            'acf_overlap': ['signal_processing', 'breath_counting', 'autocorrelation', 'overlap'],
            'overlap': ['signal_processing', 'breath_counting', 'autocorrelation', 'overlap'],
        }

        # Evaluate combinations
        if n_workers == 1:
            # Sequential execution
            results = []
            for idx, combination in enumerate(combinations, 1):
                params = dict(zip(param_names, combination))
                result = _evaluate_single_param_combination(
                    params, self.cached_signals, self.base_config, param_mapping
                )
                results.append(result)

                if verbose and idx % max(1, total_combinations // 10) == 0:
                    progress = (idx / total_combinations) * 100
                    print(f"  Progress: {idx}/{total_combinations} ({progress:.1f}%)")
        else:
            # Parallel execution
            param_dicts = [dict(zip(param_names, combo)) for combo in combinations]

            # Create partial function with fixed arguments
            eval_func = partial(
                _evaluate_single_param_combination,
                cached_signals=self.cached_signals,
                base_config=self.base_config,
                param_mapping=param_mapping
            )

            # Use multiprocessing pool
            with Pool(processes=n_workers) as pool:
                if verbose:
                    print(f"Starting parallel evaluation with {n_workers} workers...")
                    print(f"This may take a few moments to initialize...\n")

                # Map parameter combinations to workers
                results = pool.map(eval_func, param_dicts)

                if verbose:
                    print(f"✓ Parallel evaluation complete!")

        # Store all results
        self.results = results

        # Find best result
        best_loss = float('inf')
        best_result = None

        for idx, result in enumerate(results, 1):
            if result['loss'] < best_loss:
                best_loss = result['loss']
                best_result = result

                if verbose:
                    print(f"\n✓ New best found! (combination {idx}/{total_combinations})")
                    print(f"  Parameters: {result['params']}")
                    print(f"  Loss: {result['loss']:.4f}")
                    print(f"  Avg error: {result['avg_error']*100:.2f}%")
                    print(f"  Std error: {result['std_error']*100:.2f}%")

        # Summary
        if verbose:
            print(f"\n{'='*70}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Best parameters:")
            for param_name, param_value in best_result['params'].items():
                print(f"  {param_name}: {param_value}")
            print(f"\nPerformance:")
            print(f"  Loss: {best_result['loss']:.4f}")
            print(f"  Average error: {best_result['avg_error']*100:.2f}%")
            print(f"  Std error: {best_result['std_error']*100:.2f}%")
            print(f"  Max error: {best_result['max_error']*100:.2f}%")
            print(f"\nPer-video results:")
            for bpm_result in best_result['bpm_results']:
                print(f"  {bpm_result['video']}: {bpm_result['detected']:.1f} BPM "
                      f"(GT: {bpm_result['ground_truth']:.1f}, error: {bpm_result['error_pct']:.1f}%)")
            print(f"{'='*70}\n")

        return best_result
