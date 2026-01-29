"""
Script to profile the breathing analysis pipeline
"""

import sys
sys.path.append('.')

from src.pipeline import BreathingAnalyzer

# Setup paths
config_path = 'configs/default.yaml'
video_path = 'data/videos/bird_sample.mp4'
output_path = 'data/results/output_annotated.mp4'

# Run pipeline
print("Starting pipeline profiling...")
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=output_path)

print("\nResults:")
print(f"Breathing rate: {results['breathing_rate_bpm']:.1f} BPM")
print(f"Confidence: {results['confidence']:.2f}")
print(f"Total frames: {results['total_frames']}")
print(f"Signal length: {results['signal_length']}")
