#!/usr/bin/env python3
"""
Image Quality Analysis Tool with GPU Acceleration and Chromatic Aberration Detection
Analyzes brightness, sharpness, vignetting, compression artifacts, noise, and chromatic aberration
for images in camera directories using parallel computing algorithms.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from skimage import feature, filters, measure
import warnings
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading
from queue import Queue
import random
warnings.filterwarnings('ignore')


class ImageQualityAnalyzer:
    def __init__(self, input_directory, output_directory, gpu_id=0, batch_size=16, num_workers=None, sample_size=1000):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.results = {}
        self.gpu_id = gpu_id
        self.use_gpu = False
        self.batch_size = batch_size
        self.num_workers = num_workers or min(cpu_count(), 8)
        self.sample_size = sample_size
        
        # GPU memory management
        self.gpu_memory_pool = []
        self.gpu_lock = threading.Lock()
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU if available
        self.initialize_gpu()
        
        # Pre-allocate GPU memory for batch processing
        if self.use_gpu:
            self.initialize_gpu_memory_pool()
    
    def initialize_gpu(self):
        """Initialize GPU for OpenCV operations"""
        try:
            # Check if CUDA is available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
                
                # Set GPU device
                if self.gpu_id < cv2.cuda.getCudaEnabledDeviceCount():
                    cv2.cuda.setDevice(self.gpu_id)
                    self.use_gpu = True
                    print(f"Using GPU {self.gpu_id} for processing")
                    print(f"Batch size: {self.batch_size}")
                    print(f"Worker threads: {self.num_workers}")
                    print(f"Sample size per camera: {self.sample_size} images")
                    
                    # Print GPU info
                    device_info = cv2.cuda.printCudaDeviceInfo(self.gpu_id)
                    
                else:
                    print(f"GPU {self.gpu_id} not available. Using CPU instead.")
                    self.use_gpu = False
            else:
                print("No CUDA-enabled GPU found. Using CPU for processing.")
                self.use_gpu = False
                
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU processing.")
            self.use_gpu = False
    
    def initialize_gpu_memory_pool(self):
        """Pre-allocate GPU memory for common image sizes"""
        if not self.use_gpu:
            return
        
        try:
            # Common image sizes for pre-allocation
            common_sizes = [(1920, 1080), (1280, 720), (640, 480), (2048, 1024)]
            
            for size in common_sizes:
                for _ in range(2):  # 2 buffers per size
                    gpu_mat = cv2.cuda_GpuMat(*size, cv2.CV_8UC3)
                    self.gpu_memory_pool.append(gpu_mat)
            
            print(f"Pre-allocated {len(self.gpu_memory_pool)} GPU memory buffers")
            
        except Exception as e:
            print(f"GPU memory pool initialization failed: {e}")
            self.gpu_memory_pool = []
    
    def get_gpu_buffer(self, rows, cols, dtype=cv2.CV_8UC3):
        """Get a GPU buffer from the memory pool or create new one"""
        if not self.use_gpu:
            return None
        
        with self.gpu_lock:
            # Try to find a suitable buffer from the pool
            for i, buffer in enumerate(self.gpu_memory_pool):
                if buffer.rows >= rows and buffer.cols >= cols:
                    # Remove from pool and return
                    return self.gpu_memory_pool.pop(i)
        
        # Create new buffer if none available
        try:
            return cv2.cuda_GpuMat(rows, cols, dtype)
        except Exception:
            return None
    
    def return_gpu_buffer(self, buffer):
        """Return GPU buffer to the memory pool"""
        if not self.use_gpu or buffer is None:
            return
        
        with self.gpu_lock:
            if len(self.gpu_memory_pool) < 20:  # Limit pool size
                self.gpu_memory_pool.append(buffer)
    
    def process_batch_gpu(self, image_batch):
        """Process a batch of images on GPU for maximum parallelization"""
        if not self.use_gpu or not image_batch:
            return []
        
        try:
            batch_results = []
            
            # Process all images in the batch simultaneously
            gpu_images = []
            gpu_grays = []
            
            # Upload entire batch to GPU
            for image_data in image_batch:
                image, filename = image_data
                gpu_buffer = self.get_gpu_buffer(image.shape[0], image.shape[1], cv2.CV_8UC3)
                
                if gpu_buffer is not None:
                    gpu_buffer.upload(image)
                    gpu_images.append((gpu_buffer, filename))
                    
                    # Convert to grayscale
                    if len(image.shape) == 3:
                        gpu_gray = cv2.cuda.cvtColor(gpu_buffer, cv2.COLOR_BGR2GRAY)
                    else:
                        gpu_gray = gpu_buffer
                    gpu_grays.append(gpu_gray)
                else:
                    # Fallback to CPU for this image
                    gpu_images.append((image, filename))
                    gpu_grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image)
            
            # Parallel processing on GPU using CUDA streams
            streams = [cv2.cuda.Stream() for _ in range(min(len(gpu_images), 4))]
            
            for i, ((gpu_img, filename), gpu_gray) in enumerate(zip(gpu_images, gpu_grays)):
                stream = streams[i % len(streams)]
                
                try:
                    if isinstance(gpu_img, cv2.cuda_GpuMat):
                        metrics = self.calculate_metrics_gpu_parallel(gpu_img, gpu_gray, stream)
                    else:
                        metrics = self.calculate_metrics_cpu_fallback(gpu_img)
                    
                    metrics['filename'] = filename
                    batch_results.append(metrics)
                    
                except Exception as e:
                    print(f"Error processing {filename} in batch: {e}")
                    continue
            
            # Synchronize all streams
            for stream in streams:
                stream.waitForCompletion()
            
            # Return GPU buffers to pool
            for gpu_img, _ in gpu_images:
                if isinstance(gpu_img, cv2.cuda_GpuMat):
                    self.return_gpu_buffer(gpu_img)
            
            return batch_results
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            return []
    
    def calculate_metrics_gpu_parallel(self, gpu_image, gpu_gray, stream):
        """Calculate all metrics in parallel using GPU streams"""
        metrics = {}
        
        try:
            # Brightness - GPU parallel reduction
            brightness_future = self.calculate_brightness_gpu_async(gpu_gray, stream)
            
            # Sharpness - GPU Laplacian
            sharpness_future = self.calculate_sharpness_gpu_async(gpu_gray, stream)
            
            # Compression artifacts - GPU gradients
            artifacts_future = self.calculate_compression_artifacts_gpu_async(gpu_gray, stream)
            
            # Noise - GPU blur
            noise_future = self.calculate_noise_gpu_async(gpu_gray, stream)
            
            # Chromatic aberration - GPU color channel analysis
            chromatic_aberration_future = self.calculate_chromatic_aberration_gpu_async(gpu_image, stream)
            
            # Vignetting (CPU - not computation intensive)
            gray_cpu = gpu_gray.download() if isinstance(gpu_gray, cv2.cuda_GpuMat) else gpu_gray
            metrics['vignetting'] = self.calculate_vignetting_cpu(gray_cpu)
            
            # Wait for GPU computations to complete
            stream.waitForCompletion()
            
            # Collect results
            metrics['brightness'] = brightness_future
            metrics['sharpness'] = sharpness_future
            metrics['compression_artifacts'] = artifacts_future
            metrics['noise'] = noise_future
            metrics['chromatic_aberration'] = chromatic_aberration_future
            
            return metrics
            
        except Exception as e:
            print(f"GPU parallel metrics calculation failed: {e}")
            return self.calculate_metrics_cpu_fallback(gpu_image.download() if isinstance(gpu_image, cv2.cuda_GpuMat) else gpu_image)
    
    def calculate_brightness_gpu_async(self, gpu_gray, stream):
        """Asynchronous GPU brightness calculation"""
        try:
            # Use GPU sum with stream
            sum_result = cv2.cuda.sum(gpu_gray)
            brightness = sum_result[0] / (gpu_gray.rows * gpu_gray.cols)
            return float(brightness)
        except:
            return 0.0
    
    def calculate_sharpness_gpu_async(self, gpu_gray, stream):
        """Asynchronous GPU sharpness calculation"""
        try:
            # Apply Laplacian filter with stream
            gpu_laplacian = cv2.cuda.Laplacian(gpu_gray, cv2.CV_64F)
            
            # Calculate variance using GPU operations
            mean_val = cv2.cuda.sum(gpu_laplacian)[0] / (gpu_laplacian.rows * gpu_laplacian.cols)
            
            # For variance, we need to download (this could be optimized further)
            laplacian_cpu = gpu_laplacian.download()
            variance = np.var(laplacian_cpu)
            
            return float(variance)
        except:
            return 0.0
    
    def calculate_compression_artifacts_gpu_async(self, gpu_gray, stream):
        """Asynchronous GPU compression artifacts calculation"""
        try:
            # Calculate gradients using GPU Sobel filters with stream
            gpu_grad_x = cv2.cuda.Sobel(gpu_gray, cv2.CV_64F, 1, 0, ksize=3)
            gpu_grad_y = cv2.cuda.Sobel(gpu_gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Download gradients (could be optimized with GPU-only operations)
            grad_x = gpu_grad_x.download()
            grad_y = gpu_grad_y.download()
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return float(np.var(gradient_magnitude))
        except:
            return 0.0
    
    def calculate_noise_gpu_async(self, gpu_gray, stream):
        """Asynchronous GPU noise calculation"""
        try:
            # Apply Gaussian blur with stream
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
            
            # Download results for noise calculation
            gray_cpu = gpu_gray.download()
            blurred_cpu = gpu_blurred.download()
            
            # Calculate noise
            noise = gray_cpu.astype(np.float64) - blurred_cpu.astype(np.float64)
            return float(np.std(noise))
        except:
            return 0.0
    
    def calculate_chromatic_aberration_gpu_async(self, gpu_image, stream):
        """Asynchronous GPU chromatic aberration calculation"""
        try:
            # Split color channels on GPU
            gpu_channels = cv2.cuda.split(gpu_image)
            
            if len(gpu_channels) >= 3:
                # Get blue and red channels
                gpu_blue = gpu_channels[0]
                gpu_red = gpu_channels[2]
                
                # Apply edge detection to both channels
                gpu_blue_edges = cv2.cuda.Canny(gpu_blue, 50, 150)
                gpu_red_edges = cv2.cuda.Canny(gpu_red, 50, 150)
                
                # Download edge maps
                blue_edges = gpu_blue_edges.download()
                red_edges = gpu_red_edges.download()
                
                # Calculate chromatic aberration as difference in edge positions
                edge_diff = np.abs(blue_edges.astype(np.float32) - red_edges.astype(np.float32))
                chromatic_aberration = np.mean(edge_diff)
                
                return float(chromatic_aberration)
            else:
                return 0.0
        except:
            return 0.0
    
    def calculate_metrics_cpu_fallback(self, image):
        """CPU fallback for metric calculation"""
        return {
            'brightness': self.calculate_brightness_cpu(image),
            'sharpness': self.calculate_sharpness_cpu(image),
            'vignetting': self.calculate_vignetting_cpu(image),
            'compression_artifacts': self.calculate_compression_artifacts_cpu(image),
            'noise': self.calculate_noise_cpu(image),
            'chromatic_aberration': self.calculate_chromatic_aberration_cpu(image)
        }
    
    # CPU implementations for fallback
    def calculate_brightness_cpu(self, image):
        """CPU brightness calculation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def calculate_sharpness_cpu(self, image):
        """CPU sharpness calculation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_vignetting_cpu(self, gray_image):
        """CPU vignetting calculation"""
        if len(gray_image.shape) == 3:
            gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_image
        
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Center region (20% of image)
        center_size = min(h, w) // 10
        center_region = gray[center_h-center_size:center_h+center_size, 
                           center_w-center_size:center_w+center_size]
        center_brightness = np.mean(center_region)
        
        # Corner regions
        corner_size = min(h, w) // 20
        corners = [
            gray[:corner_size, :corner_size],  # top-left
            gray[:corner_size, -corner_size:],  # top-right
            gray[-corner_size:, :corner_size],  # bottom-left
            gray[-corner_size:, -corner_size:]  # bottom-right
        ]
        corner_brightness = np.mean([np.mean(corner) for corner in corners])
        
        return (center_brightness - corner_brightness) / center_brightness if center_brightness > 0 else 0
    
    def calculate_compression_artifacts_cpu(self, image):
        """CPU compression artifacts calculation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.var(gradient_magnitude)
    
    def calculate_noise_cpu(self, image):
        """CPU noise calculation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float64) - blurred.astype(np.float64)
        return np.std(noise)
    
    def calculate_chromatic_aberration_cpu(self, image):
        """CPU chromatic aberration calculation"""
        if len(image.shape) < 3:
            return 0.0  # Cannot calculate chromatic aberration for grayscale images
        
        # Split color channels
        blue_channel = image[:, :, 0]
        red_channel = image[:, :, 2]
        
        # Apply edge detection to both channels
        blue_edges = cv2.Canny(blue_channel, 50, 150)
        red_edges = cv2.Canny(red_channel, 50, 150)
        
        # Calculate chromatic aberration as difference in edge positions
        edge_diff = np.abs(blue_edges.astype(np.float32) - red_edges.astype(np.float32))
        chromatic_aberration = np.mean(edge_diff)
        
        return float(chromatic_aberration)
    
    def analyze_image_batch(self, image_paths_batch):
        """Analyze a batch of images efficiently"""
        if self.use_gpu:
            # Load all images in the batch
            image_batch = []
            for image_path in image_paths_batch:
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        image_batch.append((image, image_path.name))
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    continue
            
            # Process batch on GPU
            if image_batch:
                return self.process_batch_gpu(image_batch)
            else:
                return []
        else:
            # CPU processing with multiprocessing
            return self.process_batch_cpu_parallel(image_paths_batch)
    
    def process_batch_cpu_parallel(self, image_paths_batch):
        """Process batch using CPU multiprocessing"""
        results = []
        
        def process_single_image(image_path):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    return None
                
                metrics = self.calculate_metrics_cpu_fallback(image)
                metrics['filename'] = image_path.name
                return metrics
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None
        
        # Use ThreadPoolExecutor for I/O bound operations (image loading)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(process_single_image, path): path for path in image_paths_batch}
            
            for future in future_to_path:
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    
    def process_directory(self, directory_path):
        """Process a random sample of images in a directory using parallel batch processing"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []
        
        # Get all image files
        all_image_files = [f for f in directory_path.glob('*') if f.suffix.lower() in image_extensions]
        total_available = len(all_image_files)
        
        if total_available == 0:
            print(f"  No images found in {directory_path.name}")
            return results
        
        # Randomly sample images
        if total_available <= self.sample_size:
            print(f"  Found {total_available} images (using all available)")
            image_files = all_image_files
        else:
            print(f"  Found {total_available} images (randomly sampling {self.sample_size})")
            # Set random seed for reproducible results
            random.seed(42)
            image_files = random.sample(all_image_files, self.sample_size)
        
        total_images = len(image_files)
        print(f"  Processing {total_images} images...")
        
        # Process in batches for optimal GPU utilization
        start_time = time.time()
        processed_count = 0
        
        for i in range(0, total_images, self.batch_size):
            batch = image_files[i:i + self.batch_size]
            batch_start_time = time.time()
            
            # Process batch
            batch_results = self.analyze_image_batch(batch)
            results.extend(batch_results)
            
            processed_count += len(batch)
            batch_time = time.time() - batch_start_time
            
            # Progress reporting
            if self.use_gpu:
                throughput = len(batch) / batch_time if batch_time > 0 else 0
                print(f"  Batch {i//self.batch_size + 1}: {len(batch)} images in {batch_time:.2f}s "
                      f"({throughput:.1f} img/s) - Total: {processed_count}/{total_images}")
            else:
                print(f"  Progress: {processed_count}/{total_images} images processed")
        
        total_time = time.time() - start_time
        overall_throughput = total_images / total_time if total_time > 0 else 0
        print(f"  Directory completed: {total_images} images in {total_time:.2f}s "
              f"({overall_throughput:.1f} img/s)")
        
        return results
    
    def calculate_statistics(self, data, metrics):
        """Calculate average, median, and standard deviation for each metric"""
        stats = {}
        for metric in metrics:
            values = [item[metric] for item in data]
            stats[metric] = {
                'average': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'values': values
            }
        return stats
    
    def create_histogram_plots(self, stats, title, save_path):
        """Create histogram plots for all metrics"""
        metrics = ['brightness', 'sharpness', 'vignetting', 'compression_artifacts', 'noise', 'chromatic_aberration']
        metric_titles = ['Brightness', 'Sharpness', 'Vignetting', 'Compression Artifacts', 'Noise', 'Chromatic Aberration']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            values = stats[metric]['values']
            avg = stats[metric]['average']
            median = stats[metric]['median']
            std_val = stats[metric]['std']
            
            # Create histogram
            ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add vertical lines for average and median
            ax.axvline(avg, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg:.2f}')
            ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
            
            ax.set_title(f'{metric_title}\n(σ: {std_val:.2f})', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_to_csv(self, stats, filename):
        """Save statistics to CSV file"""
        data = []
        for metric in stats:
            data.append({
                'metric': metric,
                'average': stats[metric]['average'],
                'median': stats[metric]['median'],
                'std': stats[metric]['std']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def save_detailed_csv(self, data, filename):
        """Save detailed data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def save_histogram_data_to_csv(self, stats, filename, num_bins=20):
        """Save detailed histogram data in the format required to recreate histograms"""
        data = []
        metrics = ['brightness', 'sharpness', 'vignetting', 'compression_artifacts', 'noise', 'chromatic_aberration']
        
        for metric in metrics:
            values = stats[metric]['values']
            avg = stats[metric]['average']
            median = stats[metric]['median']
            std_val = stats[metric]['std']
            min_val = np.min(values)
            max_val = np.max(values)
            total_samples = len(values)
            
            # Create histogram
            hist, bin_edges = np.histogram(values, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate cumulative frequency
            cumulative_freq = np.cumsum(hist)
            
            for i in range(num_bins):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                bin_center = bin_centers[i]
                frequency = hist[i]
                frequency_density = frequency / (bin_end - bin_start) if (bin_end - bin_start) > 0 else 0
                cumulative_frequency = cumulative_freq[i]
                relative_frequency = frequency / total_samples if total_samples > 0 else 0
                
                data.append({
                    'metric': metric,
                    'average': avg,
                    'median': median,
                    'std': std_val,
                    'total_samples': total_samples,
                    'min_value': min_val,
                    'max_value': max_val,
                    'num_bins': num_bins,
                    'bin_number': i + 1,
                    'bin_center': bin_center,
                    'bin_start': bin_start,
                    'bin_end': bin_end,
                    'frequency': frequency,
                    'frequency_density': frequency_density,
                    'cumulative_frequency': cumulative_frequency,
                    'relative_frequency': relative_frequency
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def run_analysis(self):
        """Main analysis function"""
        metrics = ['brightness', 'sharpness', 'vignetting', 'compression_artifacts', 'noise', 'chromatic_aberration']
        all_camera_data = []
        
        print(f"Input directory: {self.input_directory}")
        print(f"Output directory: {self.output_directory}")
        print(f"Sample size per camera: {self.sample_size} images")
        print("=" * 50)
        
        # Process each camera directory
        for camera_dir in self.input_directory.iterdir():
            if camera_dir.is_dir() and camera_dir.name.startswith('CAM_'):
                print(f"Processing {camera_dir.name}...")
                
                # Analyze all images in the directory
                image_data = self.process_directory(camera_dir)
                if not image_data:
                    print(f"No valid images found in {camera_dir.name}")
                    continue
                
                # Calculate statistics
                stats = self.calculate_statistics(image_data, metrics)
                self.results[camera_dir.name] = stats
                
                # Create plots for this camera
                plot_path = self.output_directory / f"{camera_dir.name}_analysis.png"
                self.create_histogram_plots(stats, f"Image Quality Analysis - {camera_dir.name}", plot_path)
                
                # Save CSV for this camera
                csv_path = self.output_directory / f"{camera_dir.name}_statistics.csv"
                self.save_to_csv(stats, csv_path)
                
                # Save detailed CSV
                detailed_csv_path = self.output_directory / f"{camera_dir.name}_detailed.csv"
                self.save_detailed_csv(image_data, detailed_csv_path)
                
                # Add to overall data
                for item in image_data:
                    item['camera'] = camera_dir.name
                    all_camera_data.append(item)
        
        if not all_camera_data:
            print("No data found. Please check your directory structure.")
            print("Expected structure: input_dir/CAM_*/image_files.jpg")
            return
        
        # Create combined analysis for each metric
        print("Creating combined analysis...")
        self.create_combined_metric_plots(all_camera_data, metrics)
        
        # Create overall analysis
        print("Creating overall analysis...")
        overall_stats = self.calculate_statistics(all_camera_data, metrics)
        
        # Create overall plots
        overall_plot_path = self.output_directory / "overall_analysis.png"
        self.create_histogram_plots(overall_stats, "Overall Image Quality Analysis - All Nuscene Cameras", overall_plot_path)
        
        # Save overall CSV with detailed histogram data
        overall_csv_path = self.output_directory / "overall_statistics.csv"
        self.save_histogram_data_to_csv(overall_stats, overall_csv_path)
        
        # Save all detailed data
        all_detailed_csv_path = self.output_directory / "all_cameras_detailed.csv"
        self.save_detailed_csv(all_camera_data, all_detailed_csv_path)
        
        print("=" * 50)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_directory}")
        
        # Print summary
        print("\nOverall Summary:")
        for metric in metrics:
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Average: {overall_stats[metric]['average']:.2f}")
            print(f"  Median: {overall_stats[metric]['median']:.2f}")
            print(f"  Std Dev: {overall_stats[metric]['std']:.2f}")
    
    def create_combined_metric_plots(self, all_data, metrics):
        """Create combined plots showing distributions across all cameras for each metric"""
        cameras = list(set([item['camera'] for item in all_data]))
        colors = plt.cm.Set3(np.linspace(0, 1, len(cameras)))
        
        for metric in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            for i, camera in enumerate(cameras):
                camera_data = [item[metric] for item in all_data if item['camera'] == camera]
                ax.hist(camera_data, bins=15, alpha=0.6, label=camera, color=colors[i])
            
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()} Across All Cameras', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_directory / f"combined_{metric}_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save combined CSV for this metric
            metric_data = []
            for camera in cameras:
                camera_values = [item[metric] for item in all_data if item['camera'] == camera]
                metric_data.append({
                    'camera': camera,
                    'average': np.mean(camera_values),
                    'median': np.median(camera_values),
                    'std': np.std(camera_values)
                })
            
            df = pd.DataFrame(metric_data)
            csv_path = self.output_directory / f"combined_{metric}_statistics.csv"
            df.to_csv(csv_path, index=False)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze image quality metrics for camera directories with GPU acceleration')
    
    parser.add_argument('input_dir', 
                       help='Input directory containing CAM_* subdirectories with images')
    parser.add_argument('output_dir', 
                       help='Output directory where results will be saved')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use (default: 0)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU processing even if GPU is available')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for GPU processing (default: 16)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker threads for CPU processing (default: auto)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of images to randomly sample from each camera directory (default: 1000)')
    
    return parser.parse_args()


def validate_directories(input_dir, output_dir):
    """Validate input and output directories"""
    input_path = Path(input_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory!")
        return False
    
    # Check if there are CAM_ directories
    cam_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('CAM_')]
    if not cam_dirs:
        print(f"Warning: No CAM_* directories found in '{input_dir}'")
        print("Expected directory structure: input_dir/CAM_*/image_files.jpg")
    
    return True


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate directories
    if not validate_directories(args.input_dir, args.output_dir):
        sys.exit(1)
    
    print("Image Quality Analysis Tool with GPU Acceleration and Chromatic Aberration")
    print("=" * 80)
    
    # Determine GPU usage and parameters
    gpu_id = args.gpu if not args.no_gpu else None
    batch_size = args.batch_size if not args.no_gpu else 1
    num_workers = args.workers
    sample_size = args.sample_size
    
    # Run analysis
    if args.no_gpu:
        print("GPU processing disabled by user")
        analyzer = ImageQualityAnalyzer(args.input_dir, args.output_dir, 
                                      gpu_id=-1, batch_size=batch_size, 
                                      num_workers=num_workers, sample_size=sample_size)
        analyzer.use_gpu = False
    else:
        analyzer = ImageQualityAnalyzer(args.input_dir, args.output_dir, 
                                      gpu_id=gpu_id, batch_size=batch_size,
                                      num_workers=num_workers, sample_size=sample_size)
    
    start_time = time.time()
    analyzer.run_analysis()
    total_time = time.time() - start_time
    
    print("=" * 80)
    print(f"Total processing time: {total_time:.2f} seconds")
    if analyzer.use_gpu:
        print(f"Processing completed using GPU {analyzer.gpu_id} acceleration")
        print(f"Batch size: {analyzer.batch_size}")
    else:
        print(f"Processing completed using CPU with {analyzer.num_workers} workers")
    print("=" * 80)


if __name__ == "__main__":
    main()