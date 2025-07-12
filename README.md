<div align="center">
<img src=assets/logo.jpg width="30%"/>
</div>

**FastVideo is a unified framework for accelerated video generation.**

It features a clean, consistent API that works across popular video models, making it easier for developers to author new models and incorporate system- or kernel-level optimizations.
With FastVideo's optimizations, you can achieve more than 3x inference improvement compared to other systems.

<p align="center">
    | <a href="https://hao-ai-lab.github.io/FastVideo"><b>Documentation</b></a> | <a href="https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html"><b> Quick Start</b></a> | 🤗 <a href="https://huggingface.co/FastVideo/FastHunyuan"  target="_blank"><b>FastHunyuan</b></a>  | 🤗 <a href="https://huggingface.co/FastVideo/FastMochi-diffusers" target="_blank"><b>FastMochi</b></a> | 🟣💬 <a href="https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ" target="_blank"> <b>Slack</b> </a> |
</p>

<div align="center">
<img src=assets/perf.png width="90%"/>
</div>

## NEWS
- ```2025/06/14```: Release finetuning and inference code for [VSA](https://arxiv.org/pdf/2505.13389)
- ```2025/04/24```: [FastVideo V1](https://hao-ai-lab.github.io/blogs/fastvideo/) is released!
- ```2025/02/18```: Release the inference code for [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).

## Key Features

FastVideo has the following features:
- State-of-the-art performance optimizations for inference
  - [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507)
  - [TeaCache](https://arxiv.org/pdf/2411.19108)
  - [Sage Attention](https://arxiv.org/abs/2410.02367)
- Cutting edge models
  - Wan2.1 T2V, I2V
  - HunyuanVideo
  - FastHunyuan: consistency distilled video diffusion models for 8x inference speedup.
  - StepVideo T2V
- Distillation support
  - Recipes for video DiT, based on [PCM](https://github.com/G-U-N/Phased-Consistency-Model).
  - Support distilling/finetuning/inferencing state-of-the-art open video DiTs: 1. Mochi 2. Hunyuan.
- Scalable training with FSDP, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.
- Memory efficient finetuning with LoRA, precomputed latent, and precomputed text embeddings.

## Getting Started
We recommend using an environment manager such as `Conda` to create a clean environment:

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12
conda activate fastvideo

# Install FastVideo
pip install fastvideo
```

Please see our [docs](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html) for more detailed installation instructions.

## Inference
### Generating Your First Video
Here's a minimal example to generate a video using the default settings. Create a file called `example.py` with the following code:

```python
from fastvideo import VideoGenerator

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
```

Run the script with:

```bash
python example.py
```

For a more detailed guide, please see our [inference quick start](https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html).

### Other docs:

- [Design Overview](https://hao-ai-lab.github.io/FastVideo/design/overview.html)
- [Contribution Guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html)

## Distillation and Finetuning
- [Distillation Guide](https://hao-ai-lab.github.io/FastVideo/training/distillation.html)
- [Finetuning Guide](https://hao-ai-lab.github.io/FastVideo/training/finetune.html)

## 📑 Development Plan

<!-- - More distillation methods -->
  <!-- - [ ] Add Distribution Matching Distillation -->
- More models support
  <!-- - [ ] Add CogvideoX model -->
  - [x] Add StepVideo to V1
- Optimization features
  - [x] Teacache in V1
  - [x] SageAttention in V1
- Code updates
  - [x] V1 Configuration API
  - [ ] Support Training in V1
  <!-- - [ ] fp8 support -->
  <!-- - [ ] faster load model and save model support -->

## 🤝 Contributing

We welcome all contributions. Please check out our guide [here](https://hao-ai-lab.github.io/FastVideo/contributing/overview.html)

## Acknowledgement
We learned and reused code from the following projects:
- [PCM](https://github.com/G-U-N/Phased-Consistency-Model)
- [diffusers](https://github.com/huggingface/diffusers)
- [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)
- [xDiT](https://github.com/xdit-project/xDiT)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

We thank MBZUAI and [Anyscale](https://www.anyscale.com/) for their support throughout this project.

## Citation
If you use FastVideo for your research, please cite our paper:

```bibtex
@misc{zhang2025vsafastervideodiffusion,
      title={VSA: Faster Video Diffusion with Trainable Sparse Attention}, 
      author={Peiyuan Zhang and Haofeng Huang and Yongqi Chen and Will Lin and Zhengzhong Liu and Ion Stoica and Eric Xing and Hao Zhang},
      year={2025},
      eprint={2505.13389},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.13389}, 
}
@misc{zhang2025fastvideogenerationsliding,
      title={Fast Video Generation with Sliding Tile Attention},
      author={Peiyuan Zhang and Yongqi Chen and Runlong Su and Hangliang Ding and Ion Stoica and Zhenghong Liu and Hao Zhang},
      year={2025},
      eprint={2502.04507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.04507},
}
@misc{ding2025efficientvditefficientvideodiffusion,
      title={Efficient-vDiT: Efficient Video Diffusion Transformers With Attention Tile},
      author={Hangliang Ding and Dacheng Li and Runlong Su and Peiyuan Zhang and Zhijie Deng and Ion Stoica and Hao Zhang},
      year={2025},
      eprint={2502.06155},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.06155},
}
```

# Video Benchmark Suite

A comprehensive video quality assessment tool that provides an alternative to vbench with easier installation and similar functionality.

## Features

- **Multiple Quality Metrics**: PSNR, SSIM, VMAF, VIF, MSE, and more
- **Frame-by-frame Analysis**: Detailed per-frame quality metrics
- **Statistical Summaries**: Mean, standard deviation, min/max values
- **Visualization**: Automatic generation of quality metric plots
- **Multiple Output Formats**: JSON and CSV export
- **Easy Installation**: No complex dependencies like the original vbench
- **Flexible Usage**: Single video analysis or video comparison
- **Cross-platform**: Works on Linux, macOS, and Windows

## Installation

### Prerequisites

1. **Python 3.9 or higher**
2. **FFmpeg** (for video processing)

### Install Dependencies

```bash
# Install Python packages
pip install ffmpeg-quality-metrics opencv-python matplotlib numpy scikit-image

# Install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

### Quick Setup

```bash
# Clone or download the scripts
wget https://raw.githubusercontent.com/your-repo/video_benchmark_suite.py
wget https://raw.githubusercontent.com/your-repo/demo_video_benchmark.py

# Make executable
chmod +x video_benchmark_suite.py demo_video_benchmark.py
```

## Usage

### Basic Examples

#### Single Video Analysis
```bash
python video_benchmark_suite.py --video input.mp4
```

#### Video Comparison
```bash
python video_benchmark_suite.py --reference original.mp4 --distorted compressed.mp4
```

#### Custom Metrics and Settings
```bash
python video_benchmark_suite.py \
    --reference ref.mp4 \
    --distorted dist.mp4 \
    --metrics psnr ssim vmaf \
    --max-frames 100 \
    --output-dir results
```

### Command Line Options

```
usage: video_benchmark_suite.py [-h] [--video VIDEO] [--reference REFERENCE] 
                                [--distorted DISTORTED] [--metrics METRICS [METRICS ...]]
                                [--max-frames MAX_FRAMES] [--output-dir OUTPUT_DIR]
                                [--no-visualizations] [--verbose]

Video Benchmark Suite - Comprehensive video quality assessment tool

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         Single video file to analyze
  --reference REFERENCE Reference video file (for comparison)
  --distorted DISTORTED Distorted video file (for comparison)
  --metrics METRICS [METRICS ...]
                        Quality metrics to calculate (default: psnr ssim)
                        Choices: psnr, ssim, vmaf, vif, msad
  --max-frames MAX_FRAMES
                        Maximum number of frames to analyze (default: 100)
  --output-dir OUTPUT_DIR
                        Output directory for results (default: benchmark_results)
  --no-visualizations   Skip generating visualization plots
  --verbose, -v         Enable verbose output
```

### Available Metrics

| Metric | Description | Scale | Notes |
|--------|-------------|--------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | dB (higher is better) | Standard quality metric |
| **SSIM** | Structural Similarity Index | 0-1 (higher is better) | Perceptual quality metric |
| **VMAF** | Video Multi-Method Assessment Fusion | 0-100 (higher is better) | Netflix's perceptual metric |
| **VIF** | Visual Information Fidelity | 0-1 (higher is better) | Information-theoretic metric |
| **MSE** | Mean Squared Error | ≥0 (lower is better) | Basic pixel difference |

## Demo

Run the included demo to see the tool in action:

```bash
python demo_video_benchmark.py
```

This will:
1. Create sample videos with different quality levels
2. Run various quality assessments
3. Generate reports and visualizations
4. Show you where to find the results

## Output

### JSON Report
The tool generates detailed JSON reports with:
- Video information (resolution, fps, duration, etc.)
- Frame-by-frame quality metrics
- Statistical summaries (mean, std, min, max)
- Timestamps and metadata

Example output structure:
```json
{
  "reference_path": "reference.mp4",
  "distorted_path": "distorted.mp4",
  "reference_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 150,
    "duration": 5.0
  },
  "basic_metrics": {
    "psnr_values": [28.5, 29.1, 28.8, ...],
    "ssim_values": [0.85, 0.87, 0.86, ...],
    "statistics": {
      "psnr": {
        "mean": 28.8,
        "std": 1.2,
        "min": 26.1,
        "max": 31.5
      }
    }
  }
}
```

### Visualizations
Automatic PDF generation with:
- Quality metric plots over time
- Statistical summaries
- Video information comparison

## Comparison with vbench

| Feature | Video Benchmark Suite | Original vbench |
|---------|----------------------|-----------------|
| **Installation** | ✅ Easy (pip install) | ❌ Complex (many dependencies) |
| **Rust Dependencies** | ✅ None required | ❌ Requires Rust toolchain |
| **VMAF Support** | ✅ Via ffmpeg-quality-metrics | ✅ Built-in |
| **Custom Metrics** | ✅ Extensible | ✅ Built-in suite |
| **Visualization** | ✅ Matplotlib plots | ✅ Built-in plots |
| **Cross-platform** | ✅ Linux/macOS/Windows | ⚠️ Limited |
| **Memory Usage** | ✅ Efficient | ⚠️ Can be high |

## Advanced Usage

### Using as a Python Module

```python
from video_benchmark_suite import VideoQualityAssessment

# Initialize
vqa = VideoQualityAssessment(output_dir="my_results")

# Single video analysis
results = vqa.analyze_single_video("video.mp4", max_frames=50)

# Video comparison
results = vqa.compare_videos("ref.mp4", "dist.mp4", 
                           metrics=['psnr', 'ssim'], 
                           max_frames=100)

# Generate reports
report_file = vqa.generate_report(results)
viz_file = vqa.create_visualizations(results)
```

### Custom Analysis

```python
# Get basic video information
info = vqa.get_video_info("video.mp4")
print(f"Resolution: {info['width']}x{info['height']}")

# Calculate specific metrics
metrics = vqa.calculate_basic_metrics("ref.mp4", "dist.mp4", max_frames=10)
print(f"Average PSNR: {metrics['statistics']['psnr']['mean']:.2f} dB")
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   sudo apt install ffmpeg  # Ubuntu/Debian
   brew install ffmpeg      # macOS
   ```

2. **ffmpeg-quality-metrics not working**
   ```bash
   # Reinstall with proper PATH
   pip install --user ffmpeg-quality-metrics
   export PATH=$PATH:~/.local/bin
   ```

3. **OpenCV import error**
   ```bash
   # Install OpenCV
   pip install opencv-python
   ```

4. **Matplotlib display issues**
   ```bash
   # For headless systems
   export MPLBACKEND=Agg
   ```

### Performance Tips

- Use `--max-frames` to limit analysis for faster results
- Skip visualizations with `--no-visualizations` for batch processing
- Use lower resolution videos for initial testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **ffmpeg-quality-metrics**: For providing the VMAF and advanced metrics
- **OpenCV**: For video processing capabilities
- **scikit-image**: For image quality metrics
- **matplotlib**: For visualization generation

## Related Projects

- [ffmpeg-quality-metrics](https://github.com/slhck/ffmpeg-quality-metrics): Advanced video quality metrics
- [VMAF](https://github.com/Netflix/vmaf): Netflix's Video Multi-Method Assessment Fusion
- [vbench](https://github.com/Vchitect/VBench): Original comprehensive video benchmark (complex installation)
