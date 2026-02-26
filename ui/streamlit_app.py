#!/usr/bin/env python3
"""
Streamlit Web UI for Breathometer - Video breathing rate analysis
"""

import warnings
import logging
import os

# Suppress all warnings before any imports
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress streamlit logger warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.watcher').setLevel(logging.ERROR)

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tempfile
import traceback
from datetime import datetime
import io
import contextlib
import pandas as pd
import cv2
import yaml
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Add src to path (parent directory of ui/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import BreathingAnalyzer

# Load UI configuration
UI_CONFIG_PATH = Path(__file__).parent / 'ui_config.yaml'
with open(UI_CONFIG_PATH, 'r') as f:
    UI_CONFIG = yaml.safe_load(f)


class StreamlitOutputCapture:
    """Capture and display stdout/stderr in real-time to Streamlit"""
    def __init__(self, container, max_lines=50):
        self.container = container
        self.output = []
        self.current_line = ""
        self.max_lines = max_lines
        self.tqdm_active = False

    def write(self, text):
        """Write method for stdout/stderr redirection"""
        if not text:
            return 0

        # Handle carriage return (used by tqdm for progress bars)
        if '\r' in text:
            # Split by carriage return and keep the last part
            parts = text.split('\r')
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Last part - this is the current line
                    self.current_line = part
                elif part.strip():
                    # Not the last part and not empty - it's a complete line
                    if self.output and self.tqdm_active:
                        # Replace the last line if tqdm is active
                        self.output[-1] = part
                    else:
                        # Avoid duplicates
                        if not self.output or self.output[-1] != part:
                            self.output.append(part)
            self.tqdm_active = True
        else:
            # Normal line (with or without newline)
            if '\n' in text:
                lines = text.split('\n')
                # Add current line to first line
                lines[0] = self.current_line + lines[0]

                # Add all complete lines (all but the last empty one)
                for line in lines[:-1]:
                    if line.strip():
                        # Avoid consecutive duplicates
                        if not self.output or self.output[-1] != line:
                            self.output.append(line)

                # Last part becomes the new current line
                self.current_line = lines[-1]
                self.tqdm_active = False
            else:
                # No newline, append to current line
                self.current_line += text

        # Update display - keep only last max_lines
        display_lines = self.output[-self.max_lines:]
        if self.current_line.strip() and not self.current_line.endswith('\n'):
            display_lines = display_lines + [self.current_line]

        if display_lines:
            self.container.code('\n'.join(display_lines), language=None)

        return len(text)

    def flush(self):
        """Flush method required for file-like objects"""
        if self.current_line.strip():
            if self.output and self.tqdm_active:
                self.output[-1] = self.current_line
            else:
                self.output.append(self.current_line)
            self.current_line = ""
            self.tqdm_active = False

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        # Flush any remaining content
        self.flush()

    def get_output(self):
        return '\n'.join(self.output)


def plot_results(results):
    """Generate matplotlib figure with breathing signal and analysis results"""

    # Get config values
    plot_cfg = UI_CONFIG['plot']

    # Use GridSpec to ensure consistent widths
    fig = plt.figure(figsize=plot_cfg['figure_size'], dpi=plot_cfg['dpi'])
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=plot_cfg['gridspec']['width_ratios'],
                  wspace=plot_cfg['gridspec']['wspace'],
                  hspace=0.35)  # Add vertical spacing between plots

    # Create axes - main plots use first column, colorbar space in second column
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0])
    ]
    cbar_ax = fig.add_subplot(gs[1, 1])  # Colorbar axis for middle plot

    # 1. Breathing signal
    signal = np.array(results['breathing_signal'])
    fps = results['video_fps']
    time_signal = np.arange(len(signal)) / fps
    signal_duration = len(signal) / fps

    # Check if window estimates are available
    window_estimates = results.get('window_estimates', [])
    has_windows = len(window_estimates) > 0

    # Plot signal with optional window boundaries
    sig_cfg = plot_cfg['signal']
    axes[0].plot(time_signal, signal,
                 linewidth=sig_cfg['line_width'],
                 alpha=sig_cfg['line_alpha'],
                 color=sig_cfg['line_color'],
                 label='Raw signal')

    if has_windows:
        # Mark window boundaries
        for i, w in enumerate(window_estimates):
            color = sig_cfg['window_boundary_color_first'] if i == 0 else sig_cfg['window_boundary_color_other']
            alpha = sig_cfg['window_boundary_alpha_first'] if i == 0 else sig_cfg['window_boundary_alpha_other']
            axes[0].axvline(w['start_time'], color=color, linestyle='--', alpha=alpha, linewidth=sig_cfg['window_boundary_width'])
            if i == 0:
                axes[0].axvline(w['end_time'], color=color, linestyle='--', alpha=alpha, linewidth=sig_cfg['window_boundary_width'], label='Window boundaries')

        overlap = results.get('acf_overlap', 0)
        axes[0].set_title(f'Breathing Signal with Window Boundaries (overlap={overlap*100:.0f}%)', fontweight='bold')
    else:
        axes[0].set_title(f'Breathing Signal - Estimated Rate: {results["breathing_rate_bpm"]:.1f} BPM', fontweight='bold')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=plot_cfg['grid_alpha'])
    axes[0].set_xlim([0, signal_duration])

    # 2. BPM estimates per window (if available) or tracking status
    if has_windows:
        bpm_cfg = plot_cfg['bpm_estimates']
        fmt_cfg = UI_CONFIG['formatting']

        window_times = [(w['start_time'] + w['end_time'])/2 for w in window_estimates]
        window_bpms = [w['bpm'] for w in window_estimates]
        window_confidences = [w['confidence'] for w in window_estimates]

        scatter = axes[1].scatter(window_times, window_bpms,
                                  s=bpm_cfg['scatter_size'],
                                  c=window_confidences,
                                  cmap=bpm_cfg['scatter_cmap'],
                                  vmin=bpm_cfg['scatter_vmin'],
                                  vmax=bpm_cfg['scatter_vmax'],
                                  edgecolors=bpm_cfg['scatter_edge_color'],
                                  linewidth=bpm_cfg['scatter_edge_width'],
                                  zorder=5)
        axes[1].plot(window_times, window_bpms,
                     bpm_cfg['line_style'],
                     linewidth=bpm_cfg['line_width'],
                     alpha=bpm_cfg['line_alpha'],
                     color=bpm_cfg['line_color'])

        # Add colorbar in dedicated axis
        cbar_cfg = plot_cfg['colorbar']
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(cbar_cfg['label'], rotation=cbar_cfg['label_rotation'], labelpad=cbar_cfg['label_pad'])

        # Final estimate line
        final_bpm = results['breathing_rate_bpm']
        axes[1].axhline(final_bpm,
                        color=bpm_cfg['final_estimate_color'],
                        linestyle='-',
                        linewidth=bpm_cfg['final_estimate_width'],
                        label=f"Final: {final_bpm:{fmt_cfg['bpm']}} BPM",
                        zorder=3)

        # Mean and std bands
        mean_bpm = np.mean(window_bpms)
        std_bpm = np.std(window_bpms)
        axes[1].axhline(mean_bpm,
                        color=bpm_cfg['mean_color'],
                        linestyle=bpm_cfg['mean_line_style'],
                        linewidth=bpm_cfg['mean_line_width'],
                        alpha=bpm_cfg['mean_alpha'],
                        label=f"Mean: {mean_bpm:{fmt_cfg['bpm']}} BPM")
        axes[1].fill_between([0, signal_duration],
                              mean_bpm - std_bpm,
                              mean_bpm + std_bpm,
                              alpha=bpm_cfg['std_band_alpha'],
                              color=bpm_cfg['mean_color'],
                              label=f'±1 std ({std_bpm:{fmt_cfg["bpm"]}})')

        axes[1].set_title('BPM Estimates per Window (Color = Confidence)', fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('BPM')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=plot_cfg['grid_alpha'])
        axes[1].set_xlim([0, signal_duration])
    else:
        # Fallback to tracking status if no window data
        track_cfg = plot_cfg['tracking']
        fmt_cfg = UI_CONFIG['formatting']

        tracking = np.array(results['tracking_status'])
        time_tracking = np.arange(len(tracking)) / fps
        axes[1].plot(time_tracking, tracking,
                     color=track_cfg['line_color'],
                     linewidth=track_cfg['line_width'])
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Tracking Success')
        success_rate = np.mean(tracking) * 100
        axes[1].set_title(f'Tracking Status (Success Rate: {success_rate:{fmt_cfg["tracking_rate"]}}%)', fontweight='bold')
        axes[1].set_ylim(track_cfg['ylim_min'], track_cfg['ylim_max'])
        axes[1].grid(True, alpha=plot_cfg['grid_alpha'])
        # Hide colorbar axis when not used
        cbar_ax.axis('off')

    # 3. Confidence per window (if available) or motion/brightness
    if has_windows:
        conf_cfg = plot_cfg['confidence']
        fmt_cfg = UI_CONFIG['formatting']

        window_times = [(w['start_time'] + w['end_time'])/2 for w in window_estimates]
        window_confidences = [w['confidence'] for w in window_estimates]

        # Get window width for bar chart
        acf_window_size = results.get('acf_window_size', 10)
        acf_overlap = results.get('acf_overlap', 0.5)
        window_width = acf_window_size * (1 - acf_overlap)

        bars = axes[2].bar(window_times, window_confidences,
                           width=window_width * conf_cfg['bar_width_factor'],
                           alpha=conf_cfg['bar_alpha'],
                           color=conf_cfg['color_high'],
                           edgecolor=conf_cfg['bar_edge_color'])

        # Color bars by confidence
        for bar, conf in zip(bars, window_confidences):
            if conf < conf_cfg['threshold_low']:
                bar.set_color(conf_cfg['color_low'])
            elif conf < conf_cfg['threshold_high']:
                bar.set_color(conf_cfg['color_medium'])
            else:
                bar.set_color(conf_cfg['color_high'])

        mean_conf = np.mean(window_confidences)
        axes[2].axhline(mean_conf,
                        color=conf_cfg['mean_line_color'],
                        linestyle=conf_cfg['mean_line_style'],
                        linewidth=conf_cfg['mean_line_width'],
                        label=f'Mean: {mean_conf:{fmt_cfg["quality_metric"]}}')

        acf_min_confidence = results.get('acf_min_confidence', 0)
        if acf_min_confidence > 0:
            axes[2].axhline(acf_min_confidence,
                            color=conf_cfg['threshold_line_color'],
                            linestyle=conf_cfg['threshold_line_style'],
                            linewidth=conf_cfg['threshold_line_width'],
                            alpha=conf_cfg['threshold_line_alpha'],
                            label=f'Min threshold: {acf_min_confidence}')

        axes[2].set_title('Confidence per Window', fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Confidence')
        axes[2].set_ylim(0, conf_cfg['ylim_max'])
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=plot_cfg['grid_alpha'], axis='y')
        axes[2].set_xlim([0, signal_duration])
    else:
        # Fallback to motion/brightness if no window data
        mb_cfg = plot_cfg['motion_brightness']

        metadata = results['metadata']
        if len(metadata['motion']) > 0:
            motion = np.array(metadata['motion'])
            brightness = np.array(metadata['brightness'])
            time_metadata = np.arange(len(motion)) / fps

            ax3a = axes[2]
            ax3a.plot(time_metadata, motion, '-',
                     label='Motion',
                     alpha=mb_cfg['motion_alpha'],
                     color=mb_cfg['motion_color'])
            ax3a.set_xlabel('Time (s)')
            ax3a.set_ylabel('Motion', color=mb_cfg['motion_color'])
            ax3a.tick_params(axis='y', labelcolor=mb_cfg['motion_color'])
            ax3a.grid(True, alpha=plot_cfg['grid_alpha'])

            ax3b = ax3a.twinx()
            ax3b.plot(time_metadata, brightness, '-',
                     label='Brightness',
                     alpha=mb_cfg['brightness_alpha'],
                     color=mb_cfg['brightness_color'])
            ax3b.set_ylabel('Brightness', color=mb_cfg['brightness_color'])
            ax3b.tick_params(axis='y', labelcolor=mb_cfg['brightness_color'])

            axes[2].set_title('Motion and Brightness', fontweight='bold')

    plt.tight_layout()
    return fig


def process_batch_from_csv(csv_data, config_path):
    """
    Process videos from CSV file with video paths and optional ground truth

    Args:
        csv_data: DataFrame with columns 'video_path' and optional 'ground_truth_bpm'
        config_path: Path to config file

    Returns:
        DataFrame with results
    """
    # Initialize analyzer
    analyzer = BreathingAnalyzer(config_path=str(config_path))

    results_list = []

    for idx, row in csv_data.iterrows():
        video_path = row['video_path']
        ground_truth = row.get('ground_truth_bpm', None)

        # Check if video exists
        if not Path(video_path).exists():
            results_list.append({
                'video_name': Path(video_path).name,
                'video_path': video_path,
                'ground_truth_bpm': ground_truth,
                'breathing_rate_bpm': None,
                'confidence': None,
                'error': None,
                'processing_status': 'error',
                'error_message': f'Video file not found: {video_path}'
            })
            continue

        try:
            # Validate video can be opened
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                test_cap.release()
                raise ValueError(f"Cannot open video file: {video_path}")
            test_cap.release()

            # Process video
            results = analyzer.process_video(video_path=video_path, output_path=None)

            # Extract metrics
            result_row = {
                'video_name': Path(video_path).name,
                'video_path': video_path,
                'ground_truth_bpm': ground_truth,
                'breathing_rate_bpm': results['breathing_rate_bpm'],
                'confidence': results['confidence'],
                'frequency_hz': results['frequency_hz'],
                'signal_length_frames': results['signal_length'],
                'video_fps': results['video_fps'],
                'duration_seconds': results['signal_length'] / results['video_fps'],
                'tracking_success_rate': np.mean(results['tracking_status']),
                'processing_status': 'success',
                'error_message': None
            }

            # Calculate error if ground truth is provided
            if ground_truth is not None and not pd.isna(ground_truth):
                result_row['error'] = abs(results['breathing_rate_bpm'] - float(ground_truth))
                result_row['error_percentage'] = (result_row['error'] / float(ground_truth)) * 100
            else:
                result_row['error'] = None
                result_row['error_percentage'] = None

            # Add validation metrics if available
            if 'validation' in results and results['validation']:
                val = results['validation']
                result_row['validation_consistent'] = val.get('is_consistent', None)
                result_row['validation_cv'] = val.get('cv', None)

            # Add quality metrics if available
            if 'quality' in results:
                for key, value in results['quality'].items():
                    result_row[f'quality_{key}'] = value

            results_list.append(result_row)

            # Yield progress
            yield idx + 1, len(csv_data), result_row

        except Exception as e:
            error_row = {
                'video_name': Path(video_path).name,
                'video_path': video_path,
                'ground_truth_bpm': ground_truth,
                'breathing_rate_bpm': None,
                'confidence': None,
                'error': None,
                'processing_status': 'error',
                'error_message': str(e)
            }
            results_list.append(error_row)
            yield idx + 1, len(csv_data), error_row

    # Return final DataFrame
    yield len(csv_data), len(csv_data), pd.DataFrame(results_list)


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
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
    if 'start_processing' not in st.session_state:
        st.session_state.start_processing = False
    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = None
    if 'batch_processing' not in st.session_state:
        st.session_state.batch_processing = False
    if 'start_batch_processing' not in st.session_state:
        st.session_state.start_batch_processing = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'current_batch_files' not in st.session_state:
        st.session_state.current_batch_files = None
    if 'current_input_method' not in st.session_state:
        st.session_state.current_input_method = None

    # Custom CSS for smaller video preview and scrollable logs
    layout_cfg = UI_CONFIG['layout']
    st.markdown(f"""
        <style>
        /* Make video preview smaller */
        [data-testid="stVideo"] {{
            max-width: {layout_cfg['video_preview_max_width']};
        }}
        /* Improve spacing */
        .stButton button {{
            width: 100%;
        }}
        /* Make code blocks in expanders scrollable with max height */
        .stExpander [data-testid="stCodeBlock"] {{
            max-height: {layout_cfg['log_expander_max_height']};
            overflow-y: auto;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("🐦 Breathometer")
    st.markdown("Analyze breathing rate from videos using computer vision")

    # Create tabs
    tab1, tab2 = st.tabs(["📹 Single Video", "📊 Batch Processing"])

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

    # Option to save output video (only for single video tab)
    save_video = st.sidebar.checkbox("Save processed video", value=False)

    # TAB 1: Single Video Processing
    with tab1:
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
            fmt_cfg = UI_CONFIG['formatting']
            st.video(uploaded_file, start_time=0)
            st.caption(f"📹 {uploaded_file.name}")
            file_size_mb = uploaded_file.size / (1024*1024)
            st.caption(f"💾 Size: {file_size_mb:{fmt_cfg['file_size_mb']}} MB")

            # Check if manual mode is selected
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            detection_mode = current_config.get('detection', {}).get('mode', 'auto')
            start_frame = current_config.get('tracking', {}).get('start_frame', 0)

            # Manual ROI selection interface (only show if manual mode is active)
            if detection_mode == 'manual':
                st.markdown("---")
                st.subheader("🎯 Manual ROI Selection")
                st.info(f"Manual detection mode is enabled. You need to specify the chest ROI coordinates on frame {start_frame}.")

                # Initialize manual ROI in session state if not exists
                if 'manual_roi' not in st.session_state:
                    st.session_state.manual_roi = None
                if 'manual_roi_frame' not in st.session_state:
                    st.session_state.manual_roi_frame = None
                if 'manual_roi_video' not in st.session_state:
                    st.session_state.manual_roi_video = None

                # Only extract frame if video changed
                if st.session_state.manual_roi_video != uploaded_file.name:
                    st.session_state.manual_roi = None
                    st.session_state.manual_roi_frame = None
                    st.session_state.manual_roi_video = uploaded_file.name

                # Extract start_frame for ROI selection
                if st.session_state.manual_roi_frame is None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        cap = cv2.VideoCapture(tmp_path)
                        # Seek to start_frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        ret, frame = cap.read()
                        cap.release()

                        if ret:
                            st.session_state.manual_roi_frame = frame
                        else:
                            st.error(f"Could not extract frame {start_frame} from video")
                    except Exception as e:
                        st.error(f"Error extracting frame {start_frame}: {e}")
                    finally:
                        # Cleanup temp file
                        try:
                            Path(tmp_path).unlink()
                        except:
                            pass

                # Display ROI selection interface if frame is available
                if st.session_state.manual_roi_frame is not None:
                    frame = st.session_state.manual_roi_frame
                    frame_height, frame_width = frame.shape[:2]

                    # Selection method radio
                    selection_method = st.radio(
                        "ROI Selection Method:",
                        ["🖱️ Draw on Image", "✏️ Enter Coordinates"],
                        horizontal=True,
                        key="roi_selection_method"
                    )

                    if selection_method == "🖱️ Draw on Image":
                        # Drawable canvas method
                        st.write(f"**Draw a rectangle** on frame {start_frame} to select the chest ROI:")
                        st.caption(f"Frame dimensions: {frame_width} x {frame_height} pixels")
                        st.info("💡 Click and drag to draw a rectangle around the chest area. You can draw multiple times - only the last rectangle will be used.")

                        # Convert frame to RGB and then to PIL Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)

                        # Calculate display size (maintain aspect ratio, max width 800px)
                        max_display_width = 800
                        scale = min(1.0, max_display_width / frame_width)
                        display_width = int(frame_width * scale)
                        display_height = int(frame_height * scale)

                        # Create canvas for drawing
                        canvas_result = st_canvas(
                            fill_color="rgba(0, 255, 0, 0.2)",  # Semi-transparent green fill
                            stroke_width=3,
                            stroke_color="#00FF00",  # Green stroke
                            background_image=frame_pil,
                            update_streamlit=True,
                            height=display_height,
                            width=display_width,
                            drawing_mode="rect",
                            point_display_radius=0,
                            key="canvas_roi",
                        )

                        # Extract ROI from canvas
                        if canvas_result.json_data is not None:
                            objects = canvas_result.json_data["objects"]
                            if len(objects) > 0:
                                # Get the last drawn rectangle
                                last_rect = objects[-1]

                                # Convert from canvas coordinates to frame coordinates
                                canvas_x = last_rect["left"]
                                canvas_y = last_rect["top"]
                                canvas_w = last_rect["width"]
                                canvas_h = last_rect["height"]

                                # Scale back to original frame size
                                roi_x = int(canvas_x / scale)
                                roi_y = int(canvas_y / scale)
                                roi_w = int(canvas_w / scale)
                                roi_h = int(canvas_h / scale)

                                # Ensure within bounds
                                roi_x = max(0, min(roi_x, frame_width - 1))
                                roi_y = max(0, min(roi_y, frame_height - 1))
                                roi_w = max(10, min(roi_w, frame_width - roi_x))
                                roi_h = max(10, min(roi_h, frame_height - roi_y))

                                # Display coordinates
                                st.write(f"**Drawn ROI:** x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

                                # Save button
                                if st.button("💾 Save Drawn ROI", use_container_width=True, key="save_drawn_roi_btn"):
                                    st.session_state.manual_roi = [roi_x, roi_y, roi_w, roi_h]
                                    st.success(f"✅ ROI saved: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                                    st.rerun()
                            else:
                                st.info("👆 Draw a rectangle on the image above")

                    else:
                        # Manual coordinate entry method
                        with st.expander("✏️ Enter ROI Coordinates", expanded=True):
                            st.write(f"Enter the bounding box coordinates (x, y, width, height) in pixels for frame {start_frame}:")
                            st.caption(f"Frame dimensions: {frame_width} x {frame_height} pixels")

                            col_x, col_y = st.columns(2)
                            col_w, col_h = st.columns(2)

                            # Get current values or defaults
                            current_roi = st.session_state.manual_roi or [0, 0, 100, 100]

                            with col_x:
                                roi_x = st.number_input("X (left)", min_value=0, max_value=frame_width-1, value=int(current_roi[0]), step=1, key="roi_x")
                            with col_y:
                                roi_y = st.number_input("Y (top)", min_value=0, max_value=frame_height-1, value=int(current_roi[1]), step=1, key="roi_y")
                            with col_w:
                                roi_w = st.number_input("Width", min_value=10, max_value=frame_width, value=int(current_roi[2]), step=1, key="roi_w")
                            with col_h:
                                roi_h = st.number_input("Height", min_value=10, max_value=frame_height, value=int(current_roi[3]), step=1, key="roi_h")

                            # Save ROI button
                            if st.button("💾 Save ROI", use_container_width=True, key="save_roi_btn"):
                                st.session_state.manual_roi = [roi_x, roi_y, roi_w, roi_h]
                                st.success(f"✅ ROI saved: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                                st.rerun()

                        # Display frame with ROI overlay
                        display_frame = frame.copy()
                        if st.session_state.manual_roi is not None:
                            roi = st.session_state.manual_roi
                            x, y, w, h = [int(v) for v in roi]
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"ROI: {w}x{h}px", (x, max(y-10, 20)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            caption_text = f"Frame {start_frame} with selected ROI (x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]})"
                        else:
                            # Draw current preview even if not saved
                            x, y, w, h = roi_x, roi_y, roi_w, roi_h
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                            cv2.putText(display_frame, f"Preview: {w}x{h}px", (x, max(y-10, 20)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                            caption_text = f"Frame {start_frame} (preview - not saved yet)"

                        # Convert BGR to RGB for display
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        st.image(display_frame_rgb, caption=caption_text)

                    # Show saved ROI status
                    if st.session_state.manual_roi is not None:
                        st.success(f"✅ **Saved ROI:** x={st.session_state.manual_roi[0]}, y={st.session_state.manual_roi[1]}, w={st.session_state.manual_roi[2]}, h={st.session_state.manual_roi[3]}")
                    else:
                        st.warning("⚠️ Please specify and save ROI coordinates before processing")

    with col2:
        st.header("Analysis")

        if uploaded_file is not None:
            # Check if video changed - reset results if new video
            if st.session_state.current_video != uploaded_file.name:
                st.session_state.results = None
                st.session_state.current_video = uploaded_file.name
                st.session_state.processing = False
                st.session_state.start_processing = False
                st.session_state.processing_logs = None

            # Check if manual mode requires ROI before enabling processing
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            detection_mode = current_config.get('detection', {}).get('mode', 'auto')

            # Determine if button should be disabled
            button_disabled = st.session_state.processing
            disable_reason = None

            if detection_mode == 'manual':
                if st.session_state.get('manual_roi') is None:
                    button_disabled = True
                    disable_reason = "manual_roi_missing"

            # Button callback to set processing state
            def start_processing_callback():
                st.session_state.processing = True
                st.session_state.start_processing = True
                st.session_state.results = None

            # Process video button
            st.button(
                "🚀 Process Video",
                type="primary",
                use_container_width=True,
                disabled=button_disabled,
                on_click=start_processing_callback
            )

            # Show helpful message if disabled due to missing ROI
            if disable_reason == "manual_roi_missing":
                st.warning("⚠️ Please save the ROI before processing (see left panel)")

            if st.session_state.start_processing and st.session_state.processing:
                # Check if manual mode requires ROI
                with open(config_path, 'r') as f:
                    current_config = yaml.safe_load(f)
                detection_mode = current_config.get('detection', {}).get('mode', 'auto')

                # Validate ROI is provided for manual mode
                if detection_mode == 'manual' and st.session_state.get('manual_roi') is None:
                    st.error("⚠️ Manual mode is enabled but no ROI was specified. Please set the ROI before processing.")
                    st.session_state.processing = False
                    st.session_state.start_processing = False
                    st.rerun()

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
                    # Prepare config with manual ROI if needed
                    config_to_use = str(config_path)
                    if detection_mode == 'manual' and st.session_state.manual_roi is not None:
                        # Create temporary config with manual_roi injected
                        with open(config_path, 'r') as f:
                            config_data = yaml.safe_load(f)

                        # Inject manual_roi into detection config
                        config_data['detection']['manual_roi'] = st.session_state.manual_roi

                        # Save to temporary config file
                        tmp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
                        yaml.dump(config_data, tmp_config)
                        tmp_config.close()
                        config_to_use = tmp_config.name

                    # Initialize analyzer with real-time log capture
                    progress_container.progress(0.1, text="Initializing analyzer...")
                    batch_cfg = UI_CONFIG['batch']
                    stream_capture = StreamlitOutputCapture(log_container, max_lines=batch_cfg['max_log_lines'])

                    with stream_capture:
                        analyzer = BreathingAnalyzer(config_path=config_to_use)

                    # Process video with real-time log capture
                    progress_container.progress(0.3, text="Processing video... This may take a few minutes.")

                    with stream_capture:
                        results = analyzer.process_video(
                            video_path=tmp_input_path,
                            output_path=tmp_output_path
                        )

                    progress_container.progress(1.0, text="Processing complete!")

                    # Store logs in session state
                    st.session_state.processing_logs = stream_capture.get_output()

                    # Store results in session state
                    st.session_state.results = {
                        'data': results,
                        'video_name': uploaded_file.name,
                        'tmp_output_path': tmp_output_path if save_video and tmp_output_path and Path(tmp_output_path).exists() else None
                    }
                    st.session_state.processing = False
                    st.session_state.start_processing = False

                    st.success("✅ Processing complete!")
                    st.rerun()

                except KeyboardInterrupt:
                    st.session_state.processing = False
                    st.session_state.start_processing = False
                    st.session_state.results = None
                    st.warning("⚠️ Processing interrupted")
                    st.rerun()

                except Exception as e:
                    st.session_state.processing = False
                    st.session_state.start_processing = False
                    st.session_state.results = None
                    st.error(f"Error during processing: {e}")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
                    st.rerun()

                finally:
                    # Cleanup temporary input file
                    try:
                        Path(tmp_input_path).unlink()
                    except:
                        pass

                    # Cleanup temporary config file if created
                    if detection_mode == 'manual' and config_to_use != str(config_path):
                        try:
                            Path(config_to_use).unlink()
                        except:
                            pass

            # Display results if available
            if st.session_state.results is not None:
                results = st.session_state.results['data']
                video_name = st.session_state.results['video_name']
                tmp_output_path = st.session_state.results['tmp_output_path']

                # Display results
                st.markdown("---")
                st.subheader("📊 Results")

                # Key metrics in columns
                fmt_cfg = UI_CONFIG['formatting']
                conf_thresholds = UI_CONFIG['confidence_thresholds']

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Breathing Rate", f"{results['breathing_rate_bpm']:{fmt_cfg['bpm']}} BPM")

                with metric_cols[1]:
                    confidence = results['confidence']

                    # Show custom styled confidence only if below low threshold
                    if confidence < conf_thresholds['low_threshold']:
                        # Determine confidence level and styling
                        if confidence < conf_thresholds['critical_threshold']:
                            conf_style = conf_thresholds['critical_confidence']
                        else:
                            conf_style = conf_thresholds['low_confidence']

                        # Display confidence metric with colored text and warning
                        st.markdown(f"""
                            <div>
                                <label style="font-size: 0.875rem; font-weight: 400;">Confidence</label>
                                <div style="color: {conf_style['color']}; font-size: 2.25rem; font-weight: 600; line-height: 1.2;">
                                    {confidence:{fmt_cfg['confidence']}}
                                </div>
                                <div style="font-size: 0.875rem; margin-top: 4px;">
                                    {conf_style['message']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Use standard metric for good confidence
                        st.metric("Confidence", f"{confidence:{fmt_cfg['confidence']}}")

                with metric_cols[2]:
                    duration = results['signal_length'] / results['video_fps']
                    st.metric("Duration", f"{duration:{fmt_cfg['duration']}} sec")
                with metric_cols[3]:
                    tracking_rate = np.mean(results['tracking_status']) * 100
                    st.metric("Tracking Success", f"{tracking_rate:{fmt_cfg['tracking_rate']}}%")

                # Detailed info in expandable sections
                with st.expander("📈 Detailed Metrics"):
                    fmt_cfg = UI_CONFIG['formatting']
                    st.write(f"**Frequency:** {results['frequency_hz']:{fmt_cfg['frequency']}} Hz")
                    st.write(f"**Signal Length:** {results['signal_length']} frames")
                    st.write(f"**Video FPS:** {results['video_fps']}")

                    # Validation info
                    if 'validation' in results and results['validation']:
                        val = results['validation']
                        if 'is_consistent' in val:
                            status = "✓ Consistent" if val['is_consistent'] else "⚠ Inconsistent"
                            st.write(f"**Validation:** {status}")
                            if 'cv' in val:
                                st.write(f"**Coefficient of Variation:** {val['cv']:{fmt_cfg['cv']}}")
                            if 'mean_rate' in val:
                                st.write(f"**Mean Rate:** {val['mean_rate']:{fmt_cfg['bpm']}} BPM")

                    # Quality metrics
                    if 'quality' in results:
                        quality = results['quality']
                        # Only show the metrics dict if it exists
                        if 'metrics' in quality and isinstance(quality['metrics'], dict):
                            st.write("**Quality Metrics:**")

                            # Display metrics in a compact 2-column layout
                            metrics_items = list(quality['metrics'].items())
                            for i in range(0, len(metrics_items), 2):
                                cols = st.columns(2)
                                for j, col in enumerate(cols):
                                    if i + j < len(metrics_items):
                                        key, value = metrics_items[i + j]
                                        # Format the key nicely (replace underscores with spaces, capitalize)
                                        nice_key = key.replace('_', ' ').title()
                                        if isinstance(value, float):
                                            col.write(f"**{nice_key}:** {value:{fmt_cfg['quality_metric']}}")
                                        elif isinstance(value, tuple):
                                            # Format ranges nicely
                                            formatted_tuple = f"({value[0]:{fmt_cfg['bpm']}}, {value[1]:{fmt_cfg['bpm']}})"
                                            col.write(f"**{nice_key}:** {formatted_tuple}")
                                        else:
                                            col.write(f"**{nice_key}:** {value}")

                # Plot results
                with st.expander("📊 Signal Analysis", expanded=True):
                    fig = plot_results(results)
                    st.pyplot(fig)
                    plt.close(fig)

                # Download processed video if available
                if tmp_output_path and Path(tmp_output_path).exists():
                    with open(tmp_output_path, 'rb') as f:
                        video_bytes = f.read()

                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name=f"processed_{video_name}",
                        mime="video/mp4",
                        key=f"video_download_{video_name}"
                    )

                # Export results
                import json
                results_json = {
                    'video_name': video_name,
                    'breathing_rate_bpm': float(results['breathing_rate_bpm']),
                    'confidence': float(results['confidence']),
                    'frequency_hz': float(results['frequency_hz']),
                    'signal_length': int(results['signal_length']),
                    'video_fps': float(results['video_fps']),
                    'tracking_success_rate': float(np.mean(results['tracking_status'])),
                    'processed_at': datetime.now().isoformat(),
                }

                # Create DataFrame for CSV export
                results_df = pd.DataFrame([results_json])

                # Download buttons
                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"results_{Path(video_name).stem}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"csv_download_{video_name}"
                    )

                with download_col2:
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(results_json, indent=2),
                        file_name=f"results_{Path(video_name).stem}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"json_download_{video_name}"
                    )
        else:
            st.info("👆 Upload a video file to begin analysis")

    # TAB 2: Batch Processing
    with tab2:
        st.header("Batch Video Processing")
        st.markdown("Process multiple videos at once")

        # Check if manual mode is enabled - warn user
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        detection_mode = current_config.get('detection', {}).get('mode', 'auto')

        if detection_mode == 'manual':
            st.warning("⚠️ **Manual detection mode is currently enabled.** Batch processing with manual mode is not supported in the UI, as each video requires its own ROI selection. Please switch to an automatic detection mode (auto/rfdetr) in your config file for batch processing.")
            st.info("💡 **Tip:** If you need to batch process with pre-defined ROIs, you can add a `manual_roi` field to each video's config programmatically outside the UI.")
            st.stop()

        # Choose input method
        input_method = st.radio(
            "Select input method:",
            ["Upload Videos", "Upload CSV"],
            help="Upload videos directly or use a CSV file with paths and ground truth values"
        )

        # Clear results if input method changed
        if st.session_state.current_input_method != input_method:
            st.session_state.batch_results = None
            st.session_state.current_batch_files = None
            st.session_state.current_input_method = input_method

        uploaded_videos = None
        uploaded_csv = None
        csv_data = None

        if input_method == "Upload Videos":
            # Multi-file uploader
            uploaded_videos = st.file_uploader(
                "Choose video files",
                type=['mp4', 'avi', 'mov', 'mkv'],
                accept_multiple_files=True,
                help="Select multiple video files for batch processing"
            )

            if uploaded_videos and len(uploaded_videos) > 0:
                # Check if files changed - clear results if different
                current_file_names = sorted([f.name for f in uploaded_videos])
                if st.session_state.current_batch_files != current_file_names:
                    st.session_state.batch_results = None
                    st.session_state.current_batch_files = current_file_names

                st.info(f"📹 {len(uploaded_videos)} video(s) selected")
            elif uploaded_videos is not None and len(uploaded_videos) == 0:
                # Files were cleared
                if st.session_state.current_batch_files is not None:
                    st.session_state.batch_results = None
                    st.session_state.current_batch_files = None

        else:  # Upload CSV
            # Instructions
            with st.expander("📖 CSV Format Instructions"):
                st.markdown("""
                Your CSV file should have the following columns:
                - **video_path** (required): Full or relative path to the video file
                - **ground_truth_bpm** (optional): Ground truth breathing rate for comparison

                **Example CSV:**
                ```
                video_path,ground_truth_bpm
                /path/to/video1.mp4,15.5
                /path/to/video2.mp4,18.2
                /path/to/video3.mp4,
                ```

                You can also use a simple CSV with just video paths (no header needed if single column).
                """)

            # CSV Upload
            uploaded_csv = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="CSV file containing video paths and optional ground truth values"
            )

            # Check if CSV changed - clear results if different
            if uploaded_csv is not None:
                current_csv_name = uploaded_csv.name
                if st.session_state.current_batch_files != current_csv_name:
                    st.session_state.batch_results = None
                    st.session_state.current_batch_files = current_csv_name
            else:
                # CSV was cleared
                if st.session_state.current_batch_files is not None:
                    st.session_state.batch_results = None
                    st.session_state.current_batch_files = None

        # Process based on input method
        if input_method == "Upload Videos" and uploaded_videos and len(uploaded_videos) > 0:
            # Create temporary directory for uploaded videos
            temp_dir = tempfile.mkdtemp()
            video_paths = []

            # Save uploaded videos to temp directory
            for video_file in uploaded_videos:
                temp_path = Path(temp_dir) / video_file.name
                with open(temp_path, 'wb') as f:
                    f.write(video_file.read())
                video_paths.append(str(temp_path))

            # Create DataFrame for processing
            csv_data = pd.DataFrame({
                'video_path': video_paths
            })

            # Display preview
            st.subheader("📋 Videos to Process")
            preview_df = pd.DataFrame({
                'video_name': [Path(p).name for p in video_paths]
            })
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"Total videos: {len(csv_data)}")

            # Button callback to set batch processing state
            def start_batch_processing_callback():
                st.session_state.batch_processing = True
                st.session_state.start_batch_processing = True
                st.session_state.batch_results = None  # Clear previous results

            # Process button
            st.button(
                "🚀 Process All Videos",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.batch_processing,
                on_click=start_batch_processing_callback
            )

        elif input_method == "Upload CSV" and uploaded_csv is not None:
            try:
                # Read CSV
                csv_data = pd.read_csv(uploaded_csv)

                # Display preview
                st.subheader("📋 CSV Preview")
                st.dataframe(csv_data.head(10), use_container_width=True)
                st.caption(f"Total videos: {len(csv_data)}")

                # Validate CSV format
                if 'video_path' not in csv_data.columns:
                    # If no header, assume first column is video_path
                    if len(csv_data.columns) == 1:
                        csv_data.columns = ['video_path']
                    elif len(csv_data.columns) == 2:
                        csv_data.columns = ['video_path', 'ground_truth_bpm']
                    else:
                        st.error("CSV must have either 'video_path' column or be a simple list of paths")
                        st.stop()

                # Button callback to set batch processing state
                def start_batch_processing_callback():
                    st.session_state.batch_processing = True
                    st.session_state.start_batch_processing = True
                    st.session_state.batch_results = None  # Clear previous results

                # Process button
                st.button(
                    "🚀 Process All Videos",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.batch_processing,
                    on_click=start_batch_processing_callback
                )

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

        # Process videos when csv_data is available and processing is started
        if csv_data is not None and st.session_state.start_batch_processing and st.session_state.batch_processing:
            # Progress tracking
            progress_bar = st.progress(0.0, text="Starting batch processing...")
            status_text = st.empty()

            # Logs expander
            with st.expander("📋 Processing Logs", expanded=True):
                log_container = st.empty()

            # Storage for results
            all_results = pd.DataFrame()
            batch_cfg = UI_CONFIG['batch']
            stream_capture = StreamlitOutputCapture(log_container, max_lines=batch_cfg['max_log_lines'])

            # Process videos
            try:
                with stream_capture:
                    for progress in process_batch_from_csv(csv_data, config_path):
                        if len(progress) == 3:
                            current, total, result = progress

                            # Update progress
                            progress_pct = current / total
                            progress_bar.progress(progress_pct, text=f"Processing video {current}/{total}...")

                            # Check if final result (DataFrame)
                            if isinstance(result, pd.DataFrame):
                                all_results = result
                                status_text.success(f"✅ Batch processing complete! Processed {total} videos")
                            else:
                                # Individual result
                                status_msg = f"✓ {result['video_name']}"
                                if result['processing_status'] == 'success':
                                    status_msg += f" → {result['breathing_rate_bpm']:.1f} BPM"
                                else:
                                    status_msg += f" → Error: {result['error_message']}"
                                status_text.text(status_msg)

                # Store results in session state before rerun
                st.session_state.batch_results = all_results

                # Reset batch processing state
                st.session_state.batch_processing = False
                st.session_state.start_batch_processing = False
                st.rerun()

            except KeyboardInterrupt:
                st.session_state.batch_processing = False
                st.session_state.start_batch_processing = False
                st.warning("⚠️ Batch processing interrupted")
                st.rerun()

            except Exception as e:
                st.session_state.batch_processing = False
                st.session_state.start_batch_processing = False
                st.error(f"Error during batch processing: {e}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                st.rerun()

        # Display batch results from session state (if available)
        if st.session_state.batch_results is not None and len(st.session_state.batch_results) > 0:
            all_results = st.session_state.batch_results

            st.markdown("---")
            st.subheader("📊 Batch Results")

            # Summary statistics
            successful = all_results[all_results['processing_status'] == 'success']
            failed = all_results[all_results['processing_status'] == 'error']

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Videos", len(all_results))
            with metric_cols[1]:
                st.metric("Successful", len(successful), delta=None)
            with metric_cols[2]:
                st.metric("Failed", len(failed), delta=None if len(failed) == 0 else f"-{len(failed)}")
            with metric_cols[3]:
                success_rate = (len(successful) / len(all_results)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")

            # Statistics for successful videos
            if len(successful) > 0:
                fmt_cfg = UI_CONFIG['formatting']
                st.markdown("#### Summary Statistics (Successful Videos)")
                stat_cols = st.columns(3)

                with stat_cols[0]:
                    mean_bpm = successful['breathing_rate_bpm'].mean()
                    std_bpm = successful['breathing_rate_bpm'].std()
                    st.metric("Mean Breathing Rate", f"{mean_bpm:{fmt_cfg['bpm']}} BPM", delta=f"Std: {std_bpm:{fmt_cfg['bpm']}}", delta_color="off", delta_arrow="off")

                with stat_cols[1]:
                    mean_conf = successful['confidence'].mean()
                    st.metric("Mean Confidence", f"{mean_conf:{fmt_cfg['confidence']}}")

                with stat_cols[2]:
                    mean_tracking = successful['tracking_success_rate'].mean()
                    st.metric("Mean Tracking Success", f"{mean_tracking:{fmt_cfg['confidence']}}")

                # If ground truth is available, show error statistics
                if 'error' in successful.columns and successful['error'].notna().any():
                    st.markdown("#### Error Analysis (vs Ground Truth)")
                    error_cols = st.columns(3)

                    with_gt = successful[successful['error'].notna()]
                    with error_cols[0]:
                        mae = with_gt['error'].mean()
                        st.metric("Mean Absolute Error", f"{mae:{fmt_cfg['error']}} BPM")

                    with error_cols[1]:
                        mape = with_gt['error_percentage'].mean()
                        st.metric("Mean Error %", f"{mape:{fmt_cfg['error_percentage']}}%")

                    with error_cols[2]:
                        max_error = with_gt['error'].max()
                        st.metric("Max Error", f"{max_error:{fmt_cfg['error']}} BPM")

            # Detailed results table
            with st.expander("📋 Detailed Results Table", expanded=True):
                fmt_cfg = UI_CONFIG['formatting']
                layout_cfg = UI_CONFIG['layout']

                # Select columns to display
                display_cols = ['video_name', 'breathing_rate_bpm', 'confidence',
                              'tracking_success_rate', 'processing_status']
                if 'ground_truth_bpm' in all_results.columns:
                    display_cols.insert(2, 'ground_truth_bpm')
                if 'error' in all_results.columns:
                    display_cols.insert(3, 'error')

                # Filter to available columns
                display_cols = [col for col in display_cols if col in all_results.columns]

                st.dataframe(
                    all_results[display_cols].style.format({
                        'breathing_rate_bpm': '{' + f':{fmt_cfg["bpm"]}' + '}',
                        'confidence': '{' + f':{fmt_cfg["confidence"]}' + '}',
                        'tracking_success_rate': '{' + f':{fmt_cfg["confidence"]}' + '}',
                        'ground_truth_bpm': '{' + f':{fmt_cfg["bpm"]}' + '}',
                        'error': '{' + f':{fmt_cfg["error"]}' + '}'
                    }, na_rep='N/A'),
                    use_container_width=True,
                    height=layout_cfg['results_table_height']
                )

            # Download results
            st.markdown("#### 📥 Download Results")

            # Add timestamp
            all_results_copy = all_results.copy()
            all_results_copy['processed_at'] = datetime.now().isoformat()

            # CSV download
            csv_buffer = io.StringIO()
            all_results_copy.to_csv(csv_buffer, index=False)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # JSON download
                json_str = all_results_copy.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        else:
            st.info("👆 Upload videos or a CSV file to begin batch processing")

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
