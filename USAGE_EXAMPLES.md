# Video Benchmark Suite - Usage Examples

This document provides practical examples of how to use the video benchmark suite as an alternative to vbench.

## Quick Start

### 1. Simple Video Analysis

```bash
# Analyze a single video file
python3 simple_video_benchmark.py video.mp4

# Example output:
# ============================================================
# SIMPLE VIDEO BENCHMARK
# ============================================================
# Analyzing video: video.mp4
# 
# Video: video.mp4
# Resolution: 1920x1080
# FPS: 30.00
# Duration: 10.50 seconds
# Frames analyzed: 50
# 
# Overall Statistics:
#   Mean Brightness: 127.45
#   Mean Contrast: 45.23
#   Mean Sharpness: 234.67
# 
# Analysis completed in 2.34 seconds
# Results saved to: analysis_video.json
```

### 2. Video Quality Comparison

```bash
# Compare two videos (reference vs distorted)
python3 simple_video_benchmark.py --reference original.mp4 --distorted compressed.mp4

# Example output:
# ============================================================
# SIMPLE VIDEO BENCHMARK
# ============================================================
# Comparing videos: original.mp4 vs compressed.mp4
# 
# Reference: original.mp4
# Distorted: compressed.mp4
# Frames analyzed: 50
# 
# Quality Metrics:
#   PSNR: 28.45 ± 2.12 dB
#   SSIM: 0.856 ± 0.045
#   MSE:  234.56 ± 67.89
# 
# Analysis completed in 4.67 seconds
# Results saved to: comparison_original_vs_compressed.json
```

## Advanced Usage

### 3. Custom Frame Limit

```bash
# Analyze only first 20 frames for faster processing
python3 simple_video_benchmark.py video.mp4 --max-frames 20

# Compare videos with 100 frames
python3 simple_video_benchmark.py \
    --reference ref.mp4 \
    --distorted dist.mp4 \
    --max-frames 100
```

### 4. Custom Output Files

```bash
# Specify custom output file
python3 simple_video_benchmark.py video.mp4 --output my_analysis.json

# For comparisons
python3 simple_video_benchmark.py \
    --reference ref.mp4 \
    --distorted dist.mp4 \
    --output quality_comparison.json
```

### 5. Using FFmpeg Quality Metrics (if available)

```bash
# Try to use advanced metrics via ffmpeg-quality-metrics
python3 simple_video_benchmark.py \
    --reference ref.mp4 \
    --distorted dist.mp4 \
    --try-ffmpeg-metrics
```

## Understanding the Output

### JSON Report Structure

The tool generates detailed JSON reports with the following structure:

#### Single Video Analysis
```json
{
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 315,
    "duration": 10.5
  },
  "frame_stats": [
    {
      "frame": 0,
      "mean_brightness": 127.45,
      "std_brightness": 45.23,
      "contrast": 45.23,
      "sharpness": 234.67
    },
    // ... more frames
  ],
  "overall_stats": {
    "mean_brightness": 125.34,
    "mean_contrast": 44.12,
    "mean_sharpness": 245.78
  },
  "frames_analyzed": 50,
  "analysis_time": 2.34,
  "timestamp": 1640995200.0
}
```

#### Video Comparison
```json
{
  "reference_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 315,
    "duration": 10.5
  },
  "distorted_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 315,
    "duration": 10.5
  },
  "metrics": {
    "psnr_values": [28.45, 29.12, 27.89, ...],
    "ssim_values": [0.856, 0.862, 0.851, ...],
    "mse_values": [234.56, 221.34, 245.67, ...],
    "frame_numbers": [0, 1, 2, ...]
  },
  "statistics": {
    "psnr": {
      "mean": 28.45,
      "std": 2.12,
      "min": 24.56,
      "max": 32.78
    },
    "ssim": {
      "mean": 0.856,
      "std": 0.045,
      "min": 0.789,
      "max": 0.923
    },
    "mse": {
      "mean": 234.56,
      "std": 67.89,
      "min": 156.78,
      "max": 345.67
    }
  },
  "frames_analyzed": 50,
  "analysis_time": 4.67,
  "timestamp": 1640995200.0
}
```

## Quality Metrics Explained

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0 to infinity (dB)
- **Higher is better**
- **Typical values**: 
  - 20-25 dB: Poor quality
  - 25-30 dB: Fair quality
  - 30-35 dB: Good quality
  - 35+ dB: Excellent quality

### SSIM (Structural Similarity Index)
- **Range**: 0 to 1
- **Higher is better**
- **Typical values**:
  - 0.0-0.5: Poor similarity
  - 0.5-0.7: Fair similarity
  - 0.7-0.9: Good similarity
  - 0.9-1.0: Excellent similarity

### MSE (Mean Squared Error)
- **Range**: 0 to infinity
- **Lower is better**
- **Represents**: Average squared pixel differences

## Practical Workflows

### 1. Video Compression Analysis

```bash
# Step 1: Analyze original video
python3 simple_video_benchmark.py original.mp4 --output original_analysis.json

# Step 2: Compare compressed versions
python3 simple_video_benchmark.py \
    --reference original.mp4 \
    --distorted compressed_h264.mp4 \
    --output h264_comparison.json

python3 simple_video_benchmark.py \
    --reference original.mp4 \
    --distorted compressed_h265.mp4 \
    --output h265_comparison.json

# Step 3: Compare results from JSON files
```

### 2. Batch Processing

```bash
#!/bin/bash
# batch_analyze.sh

REFERENCE="reference.mp4"
OUTPUT_DIR="quality_results"

mkdir -p "$OUTPUT_DIR"

for video in compressed_*.mp4; do
    echo "Analyzing $video..."
    python3 simple_video_benchmark.py \
        --reference "$REFERENCE" \
        --distorted "$video" \
        --output "$OUTPUT_DIR/$(basename "$video" .mp4)_quality.json"
done

echo "Batch analysis complete. Results in $OUTPUT_DIR/"
```

### 3. Quick Quality Check

```bash
# Quick 10-frame quality check
python3 simple_video_benchmark.py \
    --reference original.mp4 \
    --distorted test.mp4 \
    --max-frames 10
```

## Troubleshooting

### Common Issues

1. **Video file not found**
   ```
   Error: Video file not found: video.mp4
   ```
   - Solution: Check file path and ensure video exists

2. **Cannot open video file**
   ```
   Error: Cannot open video file video.mp4
   ```
   - Solution: Ensure video format is supported by OpenCV
   - Try converting with: `ffmpeg -i input.mp4 -c copy output.mp4`

3. **Missing dependencies**
   ```
   Missing required dependency: No module named 'cv2'
   ```
   - Solution: Install dependencies:
     ```bash
     pip install opencv-python scikit-image numpy
     ```

### Performance Tips

1. **Use fewer frames for testing**: `--max-frames 10`
2. **Process shorter clips first**: Extract with `ffmpeg -t 30 -i input.mp4 test.mp4`
3. **Check video properties first**: Use single video mode to understand characteristics

## Comparison with vbench

| Feature | Simple Video Benchmark | vbench |
|---------|----------------------|---------|
| Installation | ✅ Easy (pip install) | ❌ Complex |
| Dependencies | ✅ Minimal | ❌ Many (including Rust) |
| Speed | ✅ Fast | ⚠️ Slower |
| Metrics | ✅ PSNR, SSIM, MSE | ✅ Comprehensive suite |
| VMAF Support | ⚠️ Via ffmpeg-quality-metrics | ✅ Built-in |
| Ease of Use | ✅ Simple CLI | ⚠️ Complex configuration |

## Next Steps

1. **For basic quality assessment**: Use `simple_video_benchmark.py`
2. **For advanced metrics**: Install and use `ffmpeg-quality-metrics` with `--try-ffmpeg-metrics`
3. **For comprehensive analysis**: Consider the full `video_benchmark_suite.py` (once indentation issues are resolved)
4. **For automation**: Write scripts using the JSON output format

## Example Scripts

### Python Analysis Script
```python
import json

# Load comparison results
with open('comparison_results.json', 'r') as f:
    data = json.load(f)

# Extract quality metrics
psnr_mean = data['statistics']['psnr']['mean']
ssim_mean = data['statistics']['ssim']['mean']

print(f"Quality Assessment:")
print(f"PSNR: {psnr_mean:.2f} dB")
print(f"SSIM: {ssim_mean:.3f}")

# Quality thresholds
if psnr_mean > 30 and ssim_mean > 0.9:
    print("Quality: Excellent")
elif psnr_mean > 25 and ssim_mean > 0.8:
    print("Quality: Good")
else:
    print("Quality: Needs improvement")
```

This tool provides a practical, easy-to-install alternative to vbench for video quality assessment tasks.