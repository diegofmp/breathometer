# UI Configuration

The `ui_config.yaml` file contains internal styling and formatting parameters for the Streamlit web interface.

## Purpose

This configuration file allows you to customize:

- **Plot styling**: Colors, line widths, alphas, markers, etc.
- **Number formatting**: Decimal places for different metrics
- **Layout parameters**: Widget sizes, spacing, etc.
- **Color schemes**: Status colors, plot colors, etc.

## Usage

The configuration is automatically loaded when the Streamlit app starts. Simply edit `ui_config.yaml` and restart the Streamlit app to see the changes.

## Configuration Sections

### `plot`
Visual styling for all matplotlib plots:
- `figure_size`: Overall figure dimensions
- `grid_alpha`: Grid transparency
- `signal`: First plot styling (breathing signal)
- `bpm_estimates`: Second plot styling (BPM per window)
- `confidence`: Third plot styling (confidence bars)
- `tracking`: Fallback plot for tracking status
- `motion_brightness`: Fallback plot for motion/brightness
- `colorbar`: Colorbar styling

### `formatting`
Number format strings for all metrics:
- `bpm`: Breathing rate format (e.g., '.1f' = one decimal place)
- `confidence`: Confidence/percentage format
- `frequency`: Frequency in Hz
- `error`: Error metrics
- etc.

### `layout`
UI widget dimensions and spacing:
- `video_preview_max_width`: Maximum width of video preview
- `log_expander_max_height`: Height of log containers
- `results_table_height`: Height of batch results table

### `status`
Colors for different status indicators (success, error, warning, info)

### `confidence_thresholds`
Confidence level thresholds and visual styling:
- `low_threshold`: Confidence below this is considered low (default: 0.6)
- `critical_threshold`: Confidence below this is critical (default: 0.3)
- `low_confidence`: Styling and message for low confidence (0.3 - 0.6)
- `critical_confidence`: Styling and message for critical confidence (< 0.3)
- `good_confidence`: Styling and message for good confidence (>= 0.6)

Each confidence level has:
- `color`: Text color for the metric
- `background`: Background color for the metric box
- `border`: Border style for the metric box
- `message`: Warning/info message shown below the metric

### `batch`
Batch processing parameters:
- `max_log_lines`: Maximum number of log lines to display
- `progress_update_frequency`: How often to update progress

## Example Customizations

### Change plot colors
```yaml
plot:
  signal:
    line_color: 'darkblue'  # Change signal color from steelblue to darkblue
```

### Change number precision
```yaml
formatting:
  bpm: '.2f'  # Show 2 decimal places instead of 1
```

### Adjust confidence thresholds
```yaml
# Plot thresholds (for bar chart colors)
plot:
  confidence:
    threshold_high: 0.7  # Raise high confidence threshold from 0.6 to 0.7
    threshold_low: 0.4   # Raise low confidence threshold from 0.3 to 0.4

# Metric display thresholds (for warning messages)
confidence_thresholds:
  low_threshold: 0.7  # Show warning below 0.7 instead of 0.6
  critical_threshold: 0.4  # Critical below 0.4 instead of 0.3
```

### Customize confidence warning messages
```yaml
confidence_thresholds:
  low_confidence:
    message: '⚠️ Moderate confidence - Consider verifying results manually.'
  critical_confidence:
    message: '❌ Critical - Results unreliable. Please retake video.'
```

### Change confidence metric colors
```yaml
confidence_thresholds:
  low_confidence:
    color: 'darkorange'
    background: '#ffe5cc'
    border: '3px solid #ff8800'
```

## Notes

- This file is **internal only** - it's not exposed in the Streamlit UI
- Changes require restarting the Streamlit app
- Invalid YAML syntax will cause the app to fail on startup
- Color names should be valid matplotlib colors or hex codes
