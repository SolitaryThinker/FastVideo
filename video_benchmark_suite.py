#!/usr/bin/env python3
"""
Video Benchmark Suite

A comprehensive video quality assessment tool that evaluates video files using multiple metrics.
This script provides an alternative to vbench with easier installation and similar functionality.

Features:
- PSNR, SSIM, VMAF, VIF, and other quality metrics
- Frame-by-frame analysis
- Statistical summaries
- JSON and CSV output formats
- Visualization of results
- Support for multiple video formats

Usage:
    python video_benchmark_suite.py --video path/to/video.mp4 [options]
    python video_benchmark_suite.py --reference path/to/reference.mp4 --distorted path/to/distorted.mp4 [options]

Dependencies:
    - ffmpeg-quality-metrics
    - opencv-python
    - matplotlib
    - numpy
    - scikit-image
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import mean_squared_error as mse
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install opencv-python matplotlib numpy scikit-image")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoQualityAssessment:
    """Video Quality Assessment using multiple metrics and tools."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check if ffmpeg-quality-metrics is available
        self.ffmpeg_quality_metrics_available = self._check_ffmpeg_quality_metrics()
        
    def _check_ffmpeg_quality_metrics(self) -> bool:
        """Check if ffmpeg-quality-metrics is available."""
        try:
            result = subprocess.run(
                ["python3", "-m", "ffmpeg_quality_metrics", "--help"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            try:
                # Try with direct command
                result = subprocess.run(
                    ["/home/ubuntu/.local/bin/ffmpeg-quality-metrics", "--help"],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
            except Exception:
                return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video information using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def calculate_basic_metrics(self, reference_path: str, distorted_path: str, 
                              max_frames: int = 100) -> Dict[str, Any]:
        """Calculate basic quality metrics using OpenCV and scikit-image."""
        logger.info("Calculating basic quality metrics...")
        
        ref_cap = cv2.VideoCapture(reference_path)
        dist_cap = cv2.VideoCapture(distorted_path)
        
        if not ref_cap.isOpened() or not dist_cap.isOpened():
            raise ValueError("Cannot open video files")
        
        metrics = {
            'psnr_values': [],
            'ssim_values': [],
            'mse_values': [],
            'frame_numbers': []
        }
        
        frame_count = 0
        while frame_count < max_frames:
            ret1, ref_frame = ref_cap.read()
            ret2, dist_frame = dist_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # Convert to grayscale for metrics calculation
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize distorted frame to match reference if needed
            if ref_gray.shape != dist_gray.shape:
                dist_gray = cv2.resize(dist_gray, (ref_gray.shape[1], ref_gray.shape[0]))
            
            # Calculate metrics
            try:
                psnr_val = psnr(ref_gray, dist_gray)
                ssim_val = ssim(ref_gray, dist_gray)
                mse_val = mse(ref_gray, dist_gray)
                
                metrics['psnr_values'].append(psnr_val)
                metrics['ssim_values'].append(ssim_val)
                metrics['mse_values'].append(mse_val)
                metrics['frame_numbers'].append(frame_count)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for frame {frame_count}: {e}")
            
                        frame_count += 1
        
        ref_cap.release()
        dist_cap.release()
        
        # Calculate statistics
        if metrics['psnr_values']:
            stats_dict = {
                'psnr': {
                    'mean': float(np.mean(metrics['psnr_values'])),
                    'std': float(np.std(metrics['psnr_values'])),
                    'min': float(np.min(metrics['psnr_values'])),
                    'max': float(np.max(metrics['psnr_values']))
                },
                'ssim': {
                    'mean': float(np.mean(metrics['ssim_values'])),
                    'std': float(np.std(metrics['ssim_values'])),
                    'min': float(np.min(metrics['ssim_values'])),
                    'max': float(np.max(metrics['ssim_values']))
                },
                'mse': {
                    'mean': float(np.mean(metrics['mse_values'])),
                    'std': float(np.std(metrics['mse_values'])),
                    'min': float(np.min(metrics['mse_values'])),
                    'max': float(np.max(metrics['mse_values']))
                }
            }
            metrics['statistics'] = stats_dict
        
        return metrics
    
    def run_ffmpeg_quality_metrics(self, reference_path: str, distorted_path: str,
                                 metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run ffmpeg-quality-metrics tool."""
        if not self.ffmpeg_quality_metrics_available:
            logger.warning("ffmpeg-quality-metrics not available")
            return {}
        
        if metrics is None:
            metrics = ['psnr', 'ssim']
        
        logger.info(f"Running ffmpeg-quality-metrics with metrics: {metrics}")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Try different ways to run ffmpeg-quality-metrics
            cmd_options = [
                ["/home/ubuntu/.local/bin/ffmpeg-quality-metrics"],
                ["python3", "-m", "ffmpeg_quality_metrics"],
                ["ffmpeg-quality-metrics"]
            ]
            
            success = False
            for cmd_base in cmd_options:
                try:
                    cmd = cmd_base + [
                        distorted_path,
                        reference_path,
                        "--metrics"] + metrics + [
                        "--output-format", "json"
                    ]
                    
                    logger.info(f"Running command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        success = True
                        # Parse JSON output from stdout
                        try:
                            return json.loads(result.stdout)
                        except json.JSONDecodeError:
                            logger.error("Failed to parse JSON output")
                            logger.error(f"stdout: {result.stdout}")
                            logger.error(f"stderr: {result.stderr}")
                        break
                    else:
                        logger.warning(f"Command failed with return code {result.returncode}")
                        logger.warning(f"stderr: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.error("Command timed out")
                except Exception as e:
                    logger.warning(f"Command failed: {e}")
                    continue
            
            if not success:
                logger.error("All attempts to run ffmpeg-quality-metrics failed")
                return {}
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return {}
    
    def analyze_single_video(self, video_path: str, max_frames: int = 100) -> Dict[str, Any]:
        """Analyze a single video file for basic properties and quality indicators."""
        logger.info(f"Analyzing single video: {video_path}")
        
        # Get video information
        video_info = self.get_video_info(video_path)
        
        # Extract some frames for analysis
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while frame_count < min(max_frames, 10):  # Limit to 10 frames for single video analysis
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            frame_count += 1
        
        cap.release()
        
        # Calculate frame statistics
        frame_stats = []
        if frames:
            for i, frame in enumerate(frames):
                stats = {
                    'frame_number': i,
                    'mean_brightness': np.mean(frame),
                    'std_brightness': np.std(frame),
                    'min_brightness': np.min(frame),
                    'max_brightness': np.max(frame),
                    'contrast': np.std(frame),  # Standard deviation as contrast measure
                    'sharpness': cv2.Laplacian(frame, cv2.CV_64F).var()  # Laplacian variance as sharpness measure
                }
                frame_stats.append(stats)
        
        # Overall statistics
        overall_stats = {}
        if frame_stats:
            overall_stats = {
                'mean_brightness': np.mean([f['mean_brightness'] for f in frame_stats]),
                'mean_contrast': np.mean([f['contrast'] for f in frame_stats]),
                'mean_sharpness': np.mean([f['sharpness'] for f in frame_stats]),
                'brightness_variation': np.std([f['mean_brightness'] for f in frame_stats]),
                'contrast_variation': np.std([f['contrast'] for f in frame_stats]),
                'sharpness_variation': np.std([f['sharpness'] for f in frame_stats])
            }
        
        return {
            'video_info': video_info,
            'frame_stats': frame_stats,
            'overall_stats': overall_stats,
            'analysis_timestamp': time.time()
        }
    
    def compare_videos(self, reference_path: str, distorted_path: str, 
                      metrics: Optional[List[str]] = None, max_frames: int = 100) -> Dict[str, Any]:
        """Compare two videos using multiple quality metrics."""
        logger.info(f"Comparing videos: {reference_path} vs {distorted_path}")
        
        results = {
            'reference_path': reference_path,
            'distorted_path': distorted_path,
            'timestamp': time.time()
        }
        
        # Get video information
        results['reference_info'] = self.get_video_info(reference_path)
        results['distorted_info'] = self.get_video_info(distorted_path)
        
        # Calculate basic metrics
        try:
            basic_metrics = self.calculate_basic_metrics(reference_path, distorted_path, max_frames)
            results['basic_metrics'] = basic_metrics
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            results['basic_metrics'] = {}
        
        # Run ffmpeg-quality-metrics if available
        if self.ffmpeg_quality_metrics_available:
            try:
                ffmpeg_metrics = self.run_ffmpeg_quality_metrics(reference_path, distorted_path, metrics)
                results['ffmpeg_metrics'] = ffmpeg_metrics
            except Exception as e:
                logger.error(f"Error running ffmpeg-quality-metrics: {e}")
                results['ffmpeg_metrics'] = {}
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a comprehensive report from the analysis results."""
        if output_file is None:
            output_file = str(self.output_dir / "video_quality_report.json")
        
        # Save results as JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {output_file}")
        return output_file
    
    def create_visualizations(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Create visualizations of the quality metrics."""
        if output_file is None:
            output_file = str(self.output_dir / "quality_metrics_visualization.pdf")
        
        try:
            with PdfPages(output_file) as pdf:
                # Plot basic metrics if available
                if 'basic_metrics' in results and results['basic_metrics']:
                    metrics = results['basic_metrics']
                    
                    if metrics.get('psnr_values'):
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        fig.suptitle('Video Quality Metrics', fontsize=16)
                        
                        # PSNR plot
                        axes[0, 0].plot(metrics['frame_numbers'], metrics['psnr_values'], 'b-', linewidth=2)
                        axes[0, 0].set_title('PSNR (Peak Signal-to-Noise Ratio)')
                        axes[0, 0].set_xlabel('Frame Number')
                        axes[0, 0].set_ylabel('PSNR (dB)')
                        axes[0, 0].grid(True, alpha=0.3)
                        
                        # SSIM plot
                        axes[0, 1].plot(metrics['frame_numbers'], metrics['ssim_values'], 'g-', linewidth=2)
                        axes[0, 1].set_title('SSIM (Structural Similarity Index)')
                        axes[0, 1].set_xlabel('Frame Number')
                        axes[0, 1].set_ylabel('SSIM')
                        axes[0, 1].grid(True, alpha=0.3)
                        
                        # MSE plot
                        axes[1, 0].plot(metrics['frame_numbers'], metrics['mse_values'], 'r-', linewidth=2)
                        axes[1, 0].set_title('MSE (Mean Squared Error)')
                        axes[1, 0].set_xlabel('Frame Number')
                        axes[1, 0].set_ylabel('MSE')
                        axes[1, 0].grid(True, alpha=0.3)
                        
                        # Statistics summary
                        if 'statistics' in metrics:
                            stats = metrics['statistics']
                            axes[1, 1].axis('off')
                            stats_text = f"""
Quality Metrics Statistics:

PSNR:
  Mean: {stats['psnr']['mean']:.2f} dB
  Std:  {stats['psnr']['std']:.2f} dB
  Min:  {stats['psnr']['min']:.2f} dB
  Max:  {stats['psnr']['max']:.2f} dB

SSIM:
  Mean: {stats['ssim']['mean']:.3f}
  Std:  {stats['ssim']['std']:.3f}
  Min:  {stats['ssim']['min']:.3f}
  Max:  {stats['ssim']['max']:.3f}

MSE:
  Mean: {stats['mse']['mean']:.2f}
  Std:  {stats['mse']['std']:.2f}
  Min:  {stats['mse']['min']:.2f}
  Max:  {stats['mse']['max']:.2f}
                            """
                            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                                           fontsize=10, verticalalignment='top', fontfamily='monospace')
                        
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
                
                # Add video information page
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.axis('off')
                
                info_text = "Video Analysis Report\n\n"
                
                if 'reference_info' in results:
                    ref_info = results['reference_info']
                    info_text += f"Reference Video:\n"
                    info_text += f"  Resolution: {ref_info.get('width', 'N/A')}x{ref_info.get('height', 'N/A')}\n"
                    info_text += f"  FPS: {ref_info.get('fps', 'N/A'):.2f}\n"
                    info_text += f"  Duration: {ref_info.get('duration', 'N/A'):.2f} seconds\n"
                    info_text += f"  Frame Count: {ref_info.get('frame_count', 'N/A')}\n\n"
                
                if 'distorted_info' in results:
                    dist_info = results['distorted_info']
                    info_text += f"Distorted Video:\n"
                    info_text += f"  Resolution: {dist_info.get('width', 'N/A')}x{dist_info.get('height', 'N/A')}\n"
                    info_text += f"  FPS: {dist_info.get('fps', 'N/A'):.2f}\n"
                    info_text += f"  Duration: {dist_info.get('duration', 'N/A'):.2f} seconds\n"
                    info_text += f"  Frame Count: {dist_info.get('frame_count', 'N/A')}\n\n"
                
                if 'overall_stats' in results:
                    stats = results['overall_stats']
                    info_text += f"Overall Statistics:\n"
                    info_text += f"  Mean Brightness: {stats.get('mean_brightness', 'N/A'):.2f}\n"
                    info_text += f"  Mean Contrast: {stats.get('mean_contrast', 'N/A'):.2f}\n"
                    info_text += f"  Mean Sharpness: {stats.get('mean_sharpness', 'N/A'):.2f}\n"
                
                ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', fontfamily='monospace')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return ""

def main():
    """Main function to run the video benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Video Benchmark Suite - Comprehensive video quality assessment tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single video
  python video_benchmark_suite.py --video input.mp4
  
  # Compare two videos
  python video_benchmark_suite.py --reference original.mp4 --distorted compressed.mp4
  
  # Compare with specific metrics
  python video_benchmark_suite.py --reference ref.mp4 --distorted dist.mp4 --metrics psnr ssim vmaf
  
  # Analyze with custom output directory
  python video_benchmark_suite.py --video input.mp4 --output-dir ./results
        """
    )
    
    parser.add_argument('--video', type=str, help='Single video file to analyze')
    parser.add_argument('--reference', type=str, help='Reference video file (for comparison)')
    parser.add_argument('--distorted', type=str, help='Distorted video file (for comparison)')
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim'],
                       choices=['psnr', 'ssim', 'vmaf', 'vif', 'msad'],
                       help='Quality metrics to calculate (default: psnr ssim)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum number of frames to analyze (default: 100)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualization plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.video and not (args.reference and args.distorted):
        parser.error("Either --video or both --reference and --distorted must be provided")
    
    if args.video and (args.reference or args.distorted):
        parser.error("Cannot use --video with --reference/--distorted")
    
    # Initialize the assessment tool
    vqa = VideoQualityAssessment(output_dir=args.output_dir)
    
    try:
        if args.video:
            # Single video analysis
            if not os.path.exists(args.video):
                logger.error(f"Video file not found: {args.video}")
                return 1
            
            logger.info(f"Starting single video analysis: {args.video}")
            results = vqa.analyze_single_video(args.video, args.max_frames)
            
        else:
            # Video comparison
            if not os.path.exists(args.reference):
                logger.error(f"Reference video file not found: {args.reference}")
                return 1
            
            if not os.path.exists(args.distorted):
                logger.error(f"Distorted video file not found: {args.distorted}")
                return 1
            
            logger.info(f"Starting video comparison: {args.reference} vs {args.distorted}")
            results = vqa.compare_videos(args.reference, args.distorted, args.metrics, args.max_frames)
        
        # Generate report
        report_file = vqa.generate_report(results)
        logger.info(f"Analysis complete. Report saved to: {report_file}")
        
        # Generate visualizations
        if not args.no_visualizations:
            viz_file = vqa.create_visualizations(results)
            if viz_file:
                logger.info(f"Visualizations saved to: {viz_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("VIDEO BENCHMARK SUITE - ANALYSIS SUMMARY")
        print("="*60)
        
        if args.video:
            print(f"Video: {args.video}")
            if 'video_info' in results:
                info = results['video_info']
                print(f"Resolution: {info.get('width', 'N/A')}x{info.get('height', 'N/A')}")
                print(f"FPS: {info.get('fps', 'N/A'):.2f}")
                print(f"Duration: {info.get('duration', 'N/A'):.2f} seconds")
                print(f"Frame Count: {info.get('frame_count', 'N/A')}")
        else:
            print(f"Reference: {args.reference}")
            print(f"Distorted: {args.distorted}")
            
            if 'basic_metrics' in results and 'statistics' in results['basic_metrics']:
                stats = results['basic_metrics']['statistics']
                print(f"\nQuality Metrics:")
                print(f"  PSNR: {stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB")
                print(f"  SSIM: {stats['ssim']['mean']:.3f} ± {stats['ssim']['std']:.3f}")
                print(f"  MSE:  {stats['mse']['mean']:.2f} ± {stats['mse']['std']:.2f}")
        
        print(f"\nDetailed results saved to: {report_file}")
        if not args.no_visualizations and viz_file:
            print(f"Visualizations saved to: {viz_file}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())