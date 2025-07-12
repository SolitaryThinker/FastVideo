#!/usr/bin/env python3
"""
Simple Video Benchmark Script

A working example of video quality assessment that takes a video path and runs
basic quality metrics on it. This provides the functionality requested as an
alternative to vbench.

Usage:
    python simple_video_benchmark.py video.mp4
    python simple_video_benchmark.py --reference ref.mp4 --distorted dist.mp4
"""

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import mean_squared_error as mse
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install opencv-python scikit-image numpy")
    sys.exit(1)

def get_video_info(video_path):
    """Get basic video information."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return info

def analyze_single_video(video_path, max_frames=50):
    """Analyze a single video file."""
    print(f"Analyzing video: {video_path}")
    
    # Get video info
    info = get_video_info(video_path)
    if not info:
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    # Analyze frames
    cap = cv2.VideoCapture(video_path)
    frame_stats = []
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame statistics
        stats = {
            'frame': frame_count,
            'mean_brightness': float(np.mean(gray)),
            'std_brightness': float(np.std(gray)),
            'contrast': float(np.std(gray)),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var())
        }
        frame_stats.append(stats)
        frame_count += 1
    
    cap.release()
    
    # Calculate overall statistics
    if frame_stats:
        overall_stats = {
            'mean_brightness': float(np.mean([f['mean_brightness'] for f in frame_stats])),
            'mean_contrast': float(np.mean([f['contrast'] for f in frame_stats])),
            'mean_sharpness': float(np.mean([f['sharpness'] for f in frame_stats]))
        }
    else:
        overall_stats = {}
    
    return {
        'video_info': info,
        'frame_stats': frame_stats,
        'overall_stats': overall_stats,
        'frames_analyzed': frame_count
    }

def compare_videos(reference_path, distorted_path, max_frames=50):
    """Compare two videos using quality metrics."""
    print(f"Comparing videos: {reference_path} vs {distorted_path}")
    
    # Get video info
    ref_info = get_video_info(reference_path)
    dist_info = get_video_info(distorted_path)
    
    if not ref_info or not dist_info:
        print("Error: Cannot open one or both video files")
        return None
    
    # Open videos
    ref_cap = cv2.VideoCapture(reference_path)
    dist_cap = cv2.VideoCapture(distorted_path)
    
    psnr_values = []
    ssim_values = []
    mse_values = []
    frame_numbers = []
    
    frame_count = 0
    while frame_count < max_frames:
        ret1, ref_frame = ref_cap.read()
        ret2, dist_frame = dist_cap.read()
        
        if not ret1 or not ret2:
            break
        
        # Convert to grayscale
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize if needed
        if ref_gray.shape != dist_gray.shape:
            dist_gray = cv2.resize(dist_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        # Calculate metrics
        try:
            psnr_val = psnr(ref_gray, dist_gray)
            ssim_val = ssim(ref_gray, dist_gray)
            mse_val = mse(ref_gray, dist_gray)
            
            psnr_values.append(float(psnr_val))
            ssim_values.append(float(ssim_val))
            mse_values.append(float(mse_val))
            frame_numbers.append(frame_count)
        except Exception as e:
            print(f"Warning: Error calculating metrics for frame {frame_count}: {e}")
        
        frame_count += 1
    
    ref_cap.release()
    dist_cap.release()
    
    # Calculate statistics
    stats = {}
    if psnr_values:
        stats = {
            'psnr': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values))
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)),
                'std': float(np.std(ssim_values)),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values))
            },
            'mse': {
                'mean': float(np.mean(mse_values)),
                'std': float(np.std(mse_values)),
                'min': float(np.min(mse_values)),
                'max': float(np.max(mse_values))
            }
        }
    
    return {
        'reference_info': ref_info,
        'distorted_info': dist_info,
        'metrics': {
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'mse_values': mse_values,
            'frame_numbers': frame_numbers
        },
        'statistics': stats,
        'frames_analyzed': frame_count
    }

def run_ffmpeg_quality_metrics(reference_path, distorted_path):
    """Try to run ffmpeg-quality-metrics if available."""
    try:
        # Try different command variations
        commands = [
            ["/home/ubuntu/.local/bin/ffmpeg-quality-metrics"],
            ["python3", "-m", "ffmpeg_quality_metrics"],
            ["ffmpeg-quality-metrics"]
        ]
        
        for cmd_base in commands:
            try:
                cmd = cmd_base + [
                    distorted_path,
                    reference_path,
                    "--metrics", "psnr", "ssim",
                    "--output-format", "json"
                ]
                
                print(f"Trying: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print("✓ ffmpeg-quality-metrics succeeded")
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        print("Warning: Could not parse ffmpeg-quality-metrics output")
                        return None
                else:
                    print(f"Command failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("Command timed out")
            except Exception as e:
                print(f"Command error: {e}")
                continue
        
        print("ffmpeg-quality-metrics not available or failed")
        return None
        
    except Exception as e:
        print(f"Error running ffmpeg-quality-metrics: {e}")
        return None

def save_results(results, output_file):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Simple Video Benchmark - Quality assessment tool",
        epilog="""
Examples:
  # Analyze single video
  python simple_video_benchmark.py video.mp4
  
  # Compare two videos  
  python simple_video_benchmark.py --reference original.mp4 --distorted compressed.mp4
        """
    )
    
    parser.add_argument('video', nargs='?', help='Video file to analyze (single video mode)')
    parser.add_argument('--reference', help='Reference video file (comparison mode)')
    parser.add_argument('--distorted', help='Distorted video file (comparison mode)')
    parser.add_argument('--max-frames', type=int, default=50, help='Maximum frames to analyze')
    parser.add_argument('--output', help='Output JSON file (default: auto-generated)')
    parser.add_argument('--try-ffmpeg-metrics', action='store_true', 
                       help='Try to use ffmpeg-quality-metrics if available')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and not (args.reference and args.distorted):
        parser.error("Provide either a single video file or both --reference and --distorted")
    
    if args.video and (args.reference or args.distorted):
        parser.error("Cannot use single video mode with comparison mode")
    
    print("=" * 60)
    print("SIMPLE VIDEO BENCHMARK")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        if args.video:
            # Single video analysis
            if not os.path.exists(args.video):
                print(f"Error: Video file not found: {args.video}")
                return 1
            
            results = analyze_single_video(args.video, args.max_frames)
            if not results:
                return 1
            
            # Set default output filename
            if not args.output:
                args.output = f"analysis_{Path(args.video).stem}.json"
            
            # Print summary
            print(f"\nVideo: {args.video}")
            info = results['video_info']
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']:.2f}")
            print(f"Duration: {info['duration']:.2f} seconds")
            print(f"Frames analyzed: {results['frames_analyzed']}")
            
            if results['overall_stats']:
                stats = results['overall_stats']
                print(f"\nOverall Statistics:")
                print(f"  Mean Brightness: {stats['mean_brightness']:.2f}")
                print(f"  Mean Contrast: {stats['mean_contrast']:.2f}")
                print(f"  Mean Sharpness: {stats['mean_sharpness']:.2f}")
        
        else:
            # Video comparison
            if not os.path.exists(args.reference):
                print(f"Error: Reference video not found: {args.reference}")
                return 1
            
            if not os.path.exists(args.distorted):
                print(f"Error: Distorted video not found: {args.distorted}")
                return 1
            
            results = compare_videos(args.reference, args.distorted, args.max_frames)
            if not results:
                return 1
            
            # Try ffmpeg-quality-metrics if requested
            if args.try_ffmpeg_metrics:
                print("\nTrying ffmpeg-quality-metrics...")
                ffmpeg_results = run_ffmpeg_quality_metrics(args.reference, args.distorted)
                if ffmpeg_results:
                    results['ffmpeg_metrics'] = ffmpeg_results
            
            # Set default output filename
            if not args.output:
                ref_name = Path(args.reference).stem
                dist_name = Path(args.distorted).stem
                args.output = f"comparison_{ref_name}_vs_{dist_name}.json"
            
            # Print summary
            print(f"\nReference: {args.reference}")
            print(f"Distorted: {args.distorted}")
            print(f"Frames analyzed: {results['frames_analyzed']}")
            
            if results['statistics']:
                stats = results['statistics']
                print(f"\nQuality Metrics:")
                print(f"  PSNR: {stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB")
                print(f"  SSIM: {stats['ssim']['mean']:.3f} ± {stats['ssim']['std']:.3f}")
                print(f"  MSE:  {stats['mse']['mean']:.2f} ± {stats['mse']['std']:.2f}")
        
        # Save results
        results['analysis_time'] = time.time() - start_time
        results['timestamp'] = time.time()
        
        output_file = save_results(results, args.output)
        
        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        print(f"Detailed results saved to: {output_file}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"Error: Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())