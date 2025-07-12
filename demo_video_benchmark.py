#!/usr/bin/env python3
"""
Demo script for Video Benchmark Suite

This script demonstrates how to use the video benchmark suite to analyze video quality.
It creates sample video files and runs various quality assessments on them.

Usage:
    python demo_video_benchmark.py
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_sample_video(output_path: str, width: int = 640, height: int = 480, 
                       duration: int = 5, fps: int = 30, quality: str = "high") -> bool:
    """Create a sample video file using FFmpeg."""
    
    # Different quality settings
    quality_settings = {
        "high": ["-crf", "18"],
        "medium": ["-crf", "28"],
        "low": ["-crf", "35"]
    }
    
    cmd = [
        "ffmpeg", "-y",  # Overwrite output file
        "-f", "lavfi",   # Use lavfi (libavfilter) input
        "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",  # Test pattern
        "-c:v", "libx264",  # H.264 codec
        "-preset", "fast",
        *quality_settings.get(quality, quality_settings["medium"]),
        "-pix_fmt", "yuv420p",
        output_path
    ]
    
    try:
        print(f"Creating sample video: {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✓ Successfully created: {output_path}")
            return True
        else:
            print(f"✗ Failed to create video: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout creating video: {output_path}")
        return False
    except Exception as e:
        print(f"✗ Error creating video: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    """Main demo function."""
    print("=" * 60)
    print("VIDEO BENCHMARK SUITE - DEMO")
    print("=" * 60)
    
    # Check prerequisites
    print("\n1. Checking prerequisites...")
    
    if not check_ffmpeg():
        print("✗ FFmpeg not found. Please install FFmpeg first.")
        print("  On Ubuntu/Debian: sudo apt install ffmpeg")
        print("  On macOS: brew install ffmpeg")
        return 1
    else:
        print("✓ FFmpeg found")
    
    # Check if our benchmark script exists
    benchmark_script = Path("video_benchmark_suite.py")
    if not benchmark_script.exists():
        print(f"✗ Benchmark script not found: {benchmark_script}")
        return 1
    else:
        print("✓ Video benchmark suite found")
    
    # Create temporary directory for demo files
    demo_dir = Path("demo_videos")
    demo_dir.mkdir(exist_ok=True)
    
    print(f"\n2. Creating sample videos in: {demo_dir}")
    
    # Create reference video (high quality)
    reference_video = demo_dir / "reference.mp4"
    if not create_sample_video(str(reference_video), quality="high"):
        print("Failed to create reference video")
        return 1
    
    # Create distorted video (lower quality)
    distorted_video = demo_dir / "distorted.mp4"
    if not create_sample_video(str(distorted_video), quality="low"):
        print("Failed to create distorted video")
        return 1
    
    # Create a different resolution video
    scaled_video = demo_dir / "scaled.mp4"
    if not create_sample_video(str(scaled_video), width=320, height=240, quality="medium"):
        print("Failed to create scaled video")
        return 1
    
    print("\n3. Running video quality analysis...")
    
    # Example 1: Single video analysis
    print("\n   Example 1: Single video analysis")
    cmd1 = [
        "python3", "video_benchmark_suite.py",
        "--video", str(reference_video),
        "--max-frames", "30",
        "--output-dir", "demo_results_single"
    ]
    
    try:
        print(f"   Running: {' '.join(cmd1)}")
        result = subprocess.run(cmd1, timeout=120)
        if result.returncode == 0:
            print("   ✓ Single video analysis completed successfully")
        else:
            print("   ✗ Single video analysis failed")
    except Exception as e:
        print(f"   ✗ Error running single video analysis: {e}")
    
    # Example 2: Video comparison
    print("\n   Example 2: Video comparison (reference vs distorted)")
    cmd2 = [
        "python3", "video_benchmark_suite.py",
        "--reference", str(reference_video),
        "--distorted", str(distorted_video),
        "--metrics", "psnr", "ssim",
        "--max-frames", "30",
        "--output-dir", "demo_results_comparison"
    ]
    
    try:
        print(f"   Running: {' '.join(cmd2)}")
        result = subprocess.run(cmd2, timeout=120)
        if result.returncode == 0:
            print("   ✓ Video comparison completed successfully")
        else:
            print("   ✗ Video comparison failed")
    except Exception as e:
        print(f"   ✗ Error running video comparison: {e}")
    
    # Example 3: Different resolution comparison
    print("\n   Example 3: Different resolution comparison")
    cmd3 = [
        "python3", "video_benchmark_suite.py",
        "--reference", str(reference_video),
        "--distorted", str(scaled_video),
        "--metrics", "psnr", "ssim",
        "--max-frames", "20",
        "--output-dir", "demo_results_resolution",
        "--verbose"
    ]
    
    try:
        print(f"   Running: {' '.join(cmd3)}")
        result = subprocess.run(cmd3, timeout=120)
        if result.returncode == 0:
            print("   ✓ Resolution comparison completed successfully")
        else:
            print("   ✗ Resolution comparison failed")
    except Exception as e:
        print(f"   ✗ Error running resolution comparison: {e}")
    
    print("\n4. Demo completed!")
    print("\nResults are saved in the following directories:")
    print("  - demo_results_single/")
    print("  - demo_results_comparison/")
    print("  - demo_results_resolution/")
    
    print("\nSample videos created in:")
    print(f"  - {demo_dir}/")
    
    print("\nYou can now:")
    print("1. Check the JSON reports for detailed metrics")
    print("2. View the PDF visualizations (if matplotlib worked)")
    print("3. Run your own analysis with:")
    print("   python3 video_benchmark_suite.py --help")
    
    print("\n" + "=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())