#!/usr/bin/env python3
"""
Streamlit Web UI for Breathometer - Video breathing rate analysis
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import traceback
from datetime import datetime
import io
import contextlib

# Add src to path (parent directory of ui/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import BreathingAnalyzer


class StreamlitOutputCapture:
    """Capture and display stdout/stderr in real-time to Streamlit"""
    def __init__(self, container):
        self.container = container
        self.output = []
        self.text_buffer = io.StringIO()

    def write(self, text):
        """Write method for stdout/stderr redirection"""
        if text and text.strip():
            self.output.append(text)
            # Update the container with accumulated output
            self.container.code('\n'.join(self.output), language=None)
        return len(text)

    def flush(self):
        """Flush method required for file-like objects"""
        pass

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def get_output(self):
        return '\n'.join(self.output)


def plot_results(results):
    """Generate matplotlib figure with breathing signal and analysis results"""

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # 1. Breathing signal
    signal = np.array(results['breathing_signal'])
    fps = results['video_fps']
    time = np.arange(len(signal)) / fps

    axes[0].plot(time, signal, 'b-', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Breathing Signal')
    axes[0].set_title(f'Breathing Signal - Estimated Rate: {results["breathing_rate_bpm"]:.1f} BPM')
    axes[0].grid(True, alpha=0.3)

    # 2. Tracking status
    tracking = np.array(results['tracking_status'])
    axes[1].plot(time, tracking, 'g-', linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Tracking Success')
    axes[1].set_title(f'Tracking Status (Success Rate: {np.mean(tracking)*100:.1f}%)')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)

    # 3. Metadata (motion, brightness)
    metadata = results['metadata']
    if len(metadata['motion']) > 0:
        ax3a = axes[2]
        ax3a.plot(time, metadata['motion'], 'r-', label='Motion', alpha=0.7)
        ax3a.set_xlabel('Time (s)')
        ax3a.set_ylabel('Motion', color='r')
        ax3a.tick_params(axis='y', labelcolor='r')
        ax3a.grid(True, alpha=0.3)

        ax3b = ax3a.twinx()
        brightness = np.array(metadata['brightness'])
        ax3b.plot(time, brightness, 'b-', label='Brightness', alpha=0.7)
        ax3b.set_ylabel('Brightness', color='b')
        ax3b.tick_params(axis='y', labelcolor='b')

        axes[2].set_title('Motion and Brightness')

    plt.tight_layout()
    return fig


def main():
    # Page config
    st.set_page_config(
        page_title="Breathometer",
        page_icon="🐦",
        layout="wide"
    )

    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Custom CSS for smaller video preview and scrollable logs
    st.markdown("""
        <style>
        /* Make video preview smaller */
        [data-testid="stVideo"] {
            max-width: 400px;
        }
        /* Improve spacing */
        .stButton button {
            width: 100%;
        }
        /* Make code blocks in expanders scrollable with max height */
        .stExpander [data-testid="stCodeBlock"] {
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("🐦 Breathometer")
    st.markdown("Upload a video to analyze breathing rate using computer vision")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # List available config files (parent directory of ui/)
    configs_dir = Path(__file__).parent.parent / 'configs'
    config_files = sorted(configs_dir.glob('*.yaml'))
    config_names = [cfg.name for cfg in config_files]

    if not config_names:
        st.error("No config files found in configs/ directory")
        return

    selected_config = st.sidebar.selectbox(
        "Select Config",
        config_names,
        index=config_names.index('default.yaml') if 'default.yaml' in config_names else 0
    )

    config_path = configs_dir / selected_config

    # Option to save output video
    save_video = st.sidebar.checkbox("Save processed video", value=False)

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for breathing rate analysis"
        )

        if uploaded_file is not None:
            # Smaller video preview
            st.video(uploaded_file, start_time=0)
            st.caption(f"📹 {uploaded_file.name}")
            st.caption(f"💾 Size: {uploaded_file.size / (1024*1024):.2f} MB")

    with col2:
        st.header("Analysis")

        if uploaded_file is not None:
            # Disable button while processing
            button_disabled = st.session_state.processing

            if st.button("🚀 Process Video", type="primary", use_container_width=True, disabled=button_disabled):
                st.session_state.processing = True

                # Create temporary files
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
                    tmp_input.write(uploaded_file.read())
                    tmp_input_path = tmp_input.name

                tmp_output_path = None
                if save_video:
                    tmp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4').name

                # Create containers for display
                progress_container = st.empty()

                # Logs expander (always visible during processing)
                with st.expander("📋 Processing Logs", expanded=True):
                    log_container = st.empty()

                try:
                    # Initialize analyzer with real-time log capture
                    progress_container.progress(0.1, text="Initializing analyzer...")
                    stream_capture = StreamlitOutputCapture(log_container)

                    with stream_capture:
                        analyzer = BreathingAnalyzer(config_path=str(config_path))

                    # Process video with real-time log capture
                    progress_container.progress(0.3, text="Processing video... This may take a few minutes.")

                    with stream_capture:
                        results = analyzer.process_video(
                            video_path=tmp_input_path,
                            output_path=tmp_output_path
                        )

                    progress_container.progress(1.0, text="Processing complete!")

                    st.success("✅ Processing complete!")
                    st.session_state.processing = False

                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")

                    # Key metrics in columns
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Breathing Rate", f"{results['breathing_rate_bpm']:.1f} BPM")
                    with metric_cols[1]:
                        st.metric("Confidence", f"{results['confidence']:.2%}")
                    with metric_cols[2]:
                        st.metric("Duration", f"{results['signal_length']/results['video_fps']:.1f} sec")
                    with metric_cols[3]:
                        tracking_rate = np.mean(results['tracking_status']) * 100
                        st.metric("Tracking Success", f"{tracking_rate:.1f}%")

                    # Detailed info in expandable sections
                    with st.expander("📈 Detailed Metrics"):
                        st.write(f"**Frequency:** {results['frequency_hz']:.3f} Hz")
                        st.write(f"**Signal Length:** {results['signal_length']} frames")
                        st.write(f"**Video FPS:** {results['video_fps']}")

                        # Validation info
                        if 'validation' in results and results['validation']:
                            val = results['validation']
                            if 'is_consistent' in val:
                                status = "✓ Consistent" if val['is_consistent'] else "⚠ Inconsistent"
                                st.write(f"**Validation:** {status}")
                                if 'cv' in val:
                                    st.write(f"**Coefficient of Variation:** {val['cv']:.2%}")
                                if 'mean_rate' in val:
                                    st.write(f"**Mean Rate:** {val['mean_rate']:.1f} BPM")

                        # Breath counts
                        if 'breath_counts' in results and results['breath_counts']:
                            st.write("**Breath Counts:**")
                            for window, data in results['breath_counts'].items():
                                if isinstance(data, dict):
                                    st.write(f"  - {window}: {data.get('count', 0)} breaths → {data.get('rate_bpm', 0):.1f} BPM")
                                else:
                                    st.write(f"  - {window}: {data} breaths")

                        # Quality metrics
                        if 'quality' in results:
                            st.write("**Quality Metrics:**")
                            for key, value in results['quality'].items():
                                if isinstance(value, float):
                                    st.write(f"  - {key}: {value:.3f}")
                                else:
                                    st.write(f"  - {key}: {value}")

                    # Plot results
                    with st.expander("📊 Signal Analysis", expanded=True):
                        fig = plot_results(results)
                        st.pyplot(fig)
                        plt.close(fig)

                    # Download processed video if available
                    if save_video and tmp_output_path and Path(tmp_output_path).exists():
                        with open(tmp_output_path, 'rb') as f:
                            video_bytes = f.read()

                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name=f"processed_{uploaded_file.name}",
                            mime="video/mp4"
                        )

                    # Export results as JSON
                    import json
                    results_json = {
                        'video_name': uploaded_file.name,
                        'breathing_rate_bpm': float(results['breathing_rate_bpm']),
                        'confidence': float(results['confidence']),
                        'frequency_hz': float(results['frequency_hz']),
                        'signal_length': int(results['signal_length']),
                        'video_fps': float(results['video_fps']),
                        'tracking_success_rate': float(np.mean(results['tracking_status'])),
                        'processed_at': datetime.now().isoformat(),
                    }

                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(results_json, indent=2),
                        file_name=f"results_{Path(uploaded_file.name).stem}.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.session_state.processing = False
                    st.error(f"Error during processing: {e}")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

                finally:
                    # Cleanup temporary files
                    try:
                        Path(tmp_input_path).unlink()
                        if tmp_output_path and Path(tmp_output_path).exists():
                            Path(tmp_output_path).unlink()
                    except:
                        pass
        else:
            st.info("👆 Upload a video file to begin analysis")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <small>Breathometer - Video Breathing Rate Analysis using Computer Vision</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
