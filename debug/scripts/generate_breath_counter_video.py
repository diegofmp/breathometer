"""
Generate video with breath counter overlay
"""

import sys
sys.path.append('.')

from src.pipeline import BreathingAnalyzer

# Setup paths
config_path = 'configs/default.yaml'
video_path = 'data/videos/bird_sample.mp4'
output_path = 'data/results/breath_counter_output.mp4'

print("="*70)
print("GENERATING VIDEO WITH BREATH COUNTER")
print("="*70)
print(f"Input video: {video_path}")
print(f"Output video: {output_path}")
print()

# Process video with output
analyzer = BreathingAnalyzer(config_path)
results = analyzer.process_video(video_path, output_path=output_path)

print()
print("="*70)
print("VIDEO GENERATION COMPLETE")
print("="*70)
print(f"Output saved to: {output_path}")
print(f"Total breaths detected: {results.get('breath_counts', {}).get('full', {}).get('count', 'N/A')}")
print(f"Breathing rate: {results['breathing_rate_bpm']:.1f} BPM")
print("="*70)
