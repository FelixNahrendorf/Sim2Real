#!/usr/bin/env python3
"""
Image Quality Analysis Tool with GPU Acceleration and Chromatic Aberration Detection
Analyzes brightness, sharpness, vignetting, compression artifacts, noise, and chromatic aberration
for images in complex directory structures using parallel computing algorithms.

Expected directory structure:
/app/code/seed4d/data_analysis/data_all_scenes/static/Town02/ClearNoon/vehicle.audi.tt/spawn_point_X/step_0/ego_vehicle/nuscenes_invisible/sensors/
Where images are named: 0_rgb.png, 1_rgb.png, 2_rgb.png, etc.
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
import glob
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
        
        # Camera mapping based on image numbering
        self.camera_mapping = {
            0: "CAM_FRONT",
            1: "CAM_FRONT_RIGHT", 
            2: "CAM_FRONT_LEFT",
            3: "CAM_BACK",
            # 4: Skip this camera
            5: "CAM_BACK_LEFT",
            6: "CAM_BACK_RIGHT"
        }
        
        # Initialize camera data storage
        self.camera_data = {camera: [] for camera in self.camera_mapping.values()}
        
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
                    print(f"Max spawn points to process: up to spawn_point_101")
                    
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
    
    def find_sensor_directories(self):
        """Find all sensor directories in the complex directory structure"""
        sensor_dirs = []
        
        # Look for directories matching the pattern
        pattern = "**/vehicle.audi.tt/spawn_point_*/step_0/ego_vehicle/nuscenes_invisible/sensors"
        
        for sensors_dir in self.input_directory.glob(pattern):
            if sensors_dir.is_dir():
                # Extract spawn point number for sorting
                spawn_point_part = None
                for part in sensors_dir.parts:
                    if part.startswith('spawn_point_'):
                        spawn_point_part = part
                        break
                
                if spawn_point_part:
                    try:
                        spawn_number = int(spawn_point_part.split('_')[-1])
                        if spawn_number <= 101:  # Only process up to spawn_point_101
                            sensor_dirs.append((spawn_number, sensors_dir))
                    except ValueError:
                        continue
        
        # Sort by spawn point number
        sensor_dirs.sort(key=lambda x: x[0])
        return sensor_dirs
    
    def process_sensor_directory(self, sensors_dir, spawn_point_num):
        """Process all camera images in a single sensors directory"""
        processed_images = 0
        
        for camera_num, camera_name in self.camera_mapping.items():
            # Look for image files with this camera number
            image_pattern = f"{camera_num}_rgb.png"
            image_path = sensors_dir / image_pattern
            
            if image_path.exists():
                try:
                    # Process single image
                    metrics = self.analyze_single_image(image_path)
                    if metrics:
                        metrics['spawn_point'] = spawn_point_num
                        metrics['camera'] = camera_name
                        metrics['filename'] = image_path.name
                        self.camera_data[camera_name].append(metrics)
                        processed_images += 1
                        
                        if processed_images % 10 == 0:
                            print(f"    Processed {processed_images} images from spawn_point_{spawn_point_num}")
                            
                except Exception as e:
                    print(f"    Error processing {image_path}: {e}")
            else:
                print(f"    Warning: {image_pattern} not found in {sensors_dir}")
        
        return processed_images
    
    def analyze_single_image(self, image_path):
        """Analyze a single image and return all quality metrics"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Use GPU if available, otherwise CPU
            if self.use_gpu:
                try:
                    # Upload to GPU
                    gpu_image = cv2.cuda_GpuMat()
                    gpu_image.upload(image)
                    
                    # Convert to grayscale on GPU
                    if len(image.shape) == 3:
                        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gpu_gray = gpu_image
                    
                    # Calculate metrics using GPU
                    stream = cv2.cuda.Stream()
                    metrics = self.calculate_metrics_gpu_parallel(gpu_image, gpu_gray, stream)
                    stream.waitForCompletion()
                    
                    return metrics
                except Exception as e:
                    print(f"    GPU processing failed for {image_path.name}, falling back to CPU: {e}")
                    # Fall back to CPU
                    pass
            
            # CPU processing
            return self.calculate_metrics_cpu_fallback(image)
            
        except Exception as e:
            print(f"    Error analyzing {image_path}: {e}")
            return None
    
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
    
    def save_overall_statistics_with_histogram_data(self, stats, filename):
        """Save overall statistics with complete histogram data for recreation"""
        data = []
        
        for metric in stats:
            values = stats[metric]['values']
            avg = stats[metric]['average']
            median = stats[metric]['median']
            std_val = stats[metric]['std']
            
            # Create histogram data (same as in plotting function)
            hist_counts, bin_edges = np.histogram(values, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Create base entry with statistics
            base_entry = {
                'metric': metric,
                'average': avg,
                'median': median,
                'std': std_val,
                'total_samples': len(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
                'num_bins': len(hist_counts)
            }
            
            # Add histogram bin data
            for i, (bin_center, count, bin_start, bin_end) in enumerate(zip(bin_centers, hist_counts, bin_edges[:-1], bin_edges[1:])):
                entry = base_entry.copy()
                entry.update({
                    'bin_number': i + 1,
                    'bin_center': bin_center,
                    'bin_start': bin_start,
                    'bin_end': bin_end,
                    'frequency': count,
                    'frequency_density': count / (bin_end - bin_start) if (bin_end - bin_start) > 0 else 0,
                    'cumulative_frequency': np.sum(hist_counts[:i+1]),
                    'relative_frequency': count / len(values) if len(values) > 0 else 0
                })
                data.append(entry)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        # Also save a summary file with just the key statistics
        summary_data = []
        for metric in stats:
            summary_data.append({
                'metric': metric,
                'average': stats[metric]['average'],
                'median': stats[metric]['median'],
                'std': stats[metric]['std'],
                'total_samples': len(stats[metric]['values']),
                'min_value': np.min(stats[metric]['values']),
                'max_value': np.max(stats[metric]['values'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        # Convert Path to string, then replace, then back to Path
        summary_filename = Path(str(filename).replace('.csv', '_summary.csv'))
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"Saved detailed histogram data to: {filename}")
        print(f"Saved summary statistics to: {summary_filename}")
    
    def save_detailed_csv(self, data, filename):
        """Save detailed data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def run_analysis(self):
        """Main analysis function for complex directory structure"""
        metrics = ['brightness', 'sharpness', 'vignetting', 'compression_artifacts', 'noise', 'chromatic_aberration']
        
        print(f"Input directory: {self.input_directory}")
        print(f"Output directory: {self.output_directory}")
        print("Looking for sensor directories...")
        print("=" * 80)
        
        # Find all sensor directories
        sensor_directories = self.find_sensor_directories()
        
        if not sensor_directories:
            print("No sensor directories found!")
            print("Expected structure: .../vehicle.audi.tt/spawn_point_*/step_0/ego_vehicle/nuscenes_invisible/sensors/")
            return
        
        print(f"Found {len(sensor_directories)} sensor directories")
        
        # Process each sensor directory
        total_processed = 0
        start_time = time.time()
        
        for spawn_point_num, sensors_dir in sensor_directories:
            print(f"Processing spawn_point_{spawn_point_num}: {sensors_dir}")
            
            processed_count = self.process_sensor_directory(sensors_dir, spawn_point_num)
            total_processed += processed_count
            
            if processed_count == 0:
                print(f"  No valid images processed from spawn_point_{spawn_point_num}")
        
        processing_time = time.time() - start_time
        print(f"\nTotal images processed: {total_processed}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Check if we have any data
        total_images = sum(len(data) for data in self.camera_data.values())
        if total_images == 0:
            print("No valid images were processed. Check your directory structure and file names.")
            return
        
        print(f"\nImages per camera:")
        for camera_name, data in self.camera_data.items():
            print(f"  {camera_name}: {len(data)} images")
        
        # Process results for each camera
        print("\nGenerating analysis results...")
        all_camera_data = []
        
        for camera_name, image_data in self.camera_data.items():
            if not image_data:
                print(f"No data for {camera_name}, skipping...")
                continue
                
            print(f"Analyzing {camera_name}...")
            
            # Calculate statistics
            stats = self.calculate_statistics(image_data, metrics)
            self.results[camera_name] = stats
            
            # Create plots for this camera
            plot_path = self.output_directory / f"{camera_name}_analysis.png"
            self.create_histogram_plots(stats, f"Image Quality Analysis - {camera_name}", plot_path)
            
            # Save CSV for this camera
            csv_path = self.output_directory / f"{camera_name}_statistics.csv"
            self.save_to_csv(stats, csv_path)
            
            # Save detailed CSV
            detailed_csv_path = self.output_directory / f"{camera_name}_detailed.csv"
            self.save_detailed_csv(image_data, detailed_csv_path)
            
            # Add to overall data
            all_camera_data.extend(image_data)
        
        if not all_camera_data:
            print("No camera data available for combined analysis")
            return
        
        # Create combined analysis for each metric
        print("Creating combined analysis...")
        self.create_combined_metric_plots(all_camera_data, metrics)
        
        # Create overall analysis
        print("Creating overall analysis...")
        overall_stats = self.calculate_statistics(all_camera_data, metrics)
        
        # Create overall plots
        overall_plot_path = self.output_directory / "overall_analysis.png"
        self.create_histogram_plots(overall_stats, "Overall Image Quality Analysis - All SEED4D Cameras", overall_plot_path)
        
        # Save overall CSV with complete histogram data
        overall_csv_path = self.output_directory / "overall_statistics.csv"
        self.save_overall_statistics_with_histogram_data(overall_stats, overall_csv_path)
        
        # Save all detailed data
        all_detailed_csv_path = self.output_directory / "all_cameras_detailed.csv"
        self.save_detailed_csv(all_camera_data, all_detailed_csv_path)
        
        print("=" * 80)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_directory}")
        
        # Print summary
        print(f"\nOverall Summary ({total_images} total images):")
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
    parser = argparse.ArgumentParser(description='Analyze image quality metrics for complex sensor directory structures')
    
    parser.add_argument('input_dir', 
                       help='Input directory containing the complex directory structure with vehicle.audi.tt subdirectories')
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
    
    # Check if there are vehicle directories
    vehicle_pattern = "**/vehicle.audi.tt"
    vehicle_dirs = list(input_path.glob(vehicle_pattern))
    if not vehicle_dirs:
        print(f"Warning: No 'vehicle.audi.tt' directories found in '{input_dir}'")
        print("Expected structure: input_dir/.../vehicle.audi.tt/spawn_point_*/step_0/ego_vehicle/nuscenes_invisible/sensors/")
    else:
        print(f"Found {len(vehicle_dirs)} vehicle.audi.tt directories")
    
    return True


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate directories
    if not validate_directories(args.input_dir, args.output_dir):
        sys.exit(1)
    
    print("Image Quality Analysis Tool for Complex Directory Structures")
    print("=" * 80)
    print("Camera mapping:")
    print("  0_rgb.png -> CAM_FRONT")
    print("  1_rgb.png -> CAM_FRONT_RIGHT")  
    print("  2_rgb.png -> CAM_FRONT_LEFT")
    print("  3_rgb.png -> CAM_BACK")
    print("  4_rgb.png -> SKIPPED")
    print("  5_rgb.png -> CAM_BACK_LEFT")
    print("  6_rgb.png -> CAM_BACK_RIGHT")
    print("=" * 80)
    
    # Determine GPU usage and parameters
    gpu_id = args.gpu if not args.no_gpu else None
    batch_size = args.batch_size if not args.no_gpu else 1
    num_workers = args.workers
    
    # Run analysis - note: sample_size parameter removed as we process all found images
    if args.no_gpu:
        print("GPU processing disabled by user")
        analyzer = ImageQualityAnalyzer(args.input_dir, args.output_dir, 
                                      gpu_id=-1, batch_size=batch_size, 
                                      num_workers=num_workers, sample_size=None)
        analyzer.use_gpu = False
    else:
        analyzer = ImageQualityAnalyzer(args.input_dir, args.output_dir, 
                                      gpu_id=gpu_id, batch_size=batch_size,
                                      num_workers=num_workers, sample_size=None)
    
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