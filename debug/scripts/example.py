#!/usr/bin/env python3
"""
Simple example script for bird breathing analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import BreathingAnalyzer


def main():
    """
    Run bird breathing analysis
    """
    print("="*60)
    print("BIRD BREATHING ANALYSIS - SIMPLE EXAMPLE")
    print("="*60)
    print()
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = BreathingAnalyzer('configs/default.yaml')
    print()
    
    # Process video
    # REPLACE THIS with your video path
    video_path = 'data/videos/bird_sample.mp4'
    output_path = 'data/results/output_annotated.mp4'
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print()
        print("Please:")
        print("1. Add your video to data/videos/")
        print("2. Update the video_path variable in this script")
        print("3. Run again")
        return
    
    # Run analysis
    try:
        results = analyzer.process_video(
            video_path,
            output_path=output_path
        )
        
        # Display results
        print()
        print("="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print()
        print(f"📊 Results:")
        print(f"  Breathing Rate: {results['breathing_rate_bpm']:.1f} BPM")
        print(f"  Confidence: {results['confidence']:.2f}")
        print(f"  Frequency: {results['frequency_hz']:.2f} Hz")
        print()
        print(f"📹 Output video: {output_path}")
        print(f"📁 Signal data: {len(results['breathing_signal'])} frames")
        print()
        
        # Save results
        import json
        results_json = 'data/results/breathing_analysis.json'
        
        export_results = {
            'breathing_rate_bpm': float(results['breathing_rate_bpm']),
            'confidence': float(results['confidence']),
            'frequency_hz': float(results['frequency_hz']),
            'video_fps': float(results['video_fps']),
            'signal_length': int(results['signal_length'])
        }
        
        with open(results_json, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_json}")
        print()
        print("="*60)
        print("✓ Done!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check that video file exists and is valid")
        print("2. Ensure bird and hand are visible in video")
        print("3. Try adjusting config in configs/default.yaml")
        print("4. Check SETUP.md for more help")


if __name__ == "__main__":
    main()
