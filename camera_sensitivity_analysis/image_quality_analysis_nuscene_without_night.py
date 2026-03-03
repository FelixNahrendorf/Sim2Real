#!/usr/bin/env python3
"""
Image Quality Analysis Tool with GPU Acceleration and Chromatic Aberration Detection
Analyzes brightness, sharpness, vignetting, compression artifacts, noise, chromatic aberration,
saturation, laplacian variance, and FFT power spectrum for images in camera directories.
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


ALL_METRICS = [
    'brightness', 'sharpness', 'vignetting', 'compression_artifacts',
    'noise', 'chromatic_aberration', 'saturation', 'laplacian_variance', 'fft_power_spectrum'
]

METRIC_TITLES = {
    'brightness': 'Brightness',
    'sharpness': 'Sharpness',
    'vignetting': 'Vignetting',
    'compression_artifacts': 'Compression Artifacts',
    'noise': 'Noise',
    'chromatic_aberration': 'Chromatic Aberration',
    'saturation': 'Saturation',
    'laplacian_variance': 'Laplacian Variance',
    'fft_power_spectrum': 'FFT Power Spectrum (High-Freq Ratio)',
}


def load_night_scene_filenames(trainval_txt, test_txt, nuscenes_dataroot):
    """
    Merge night scene sample tokens from both train/val and test txt files,
    then use the NuScenes API to resolve them to actual image filenames.
    Returns a set of image filenames (basenames) that belong to night scenes.
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        raise ImportError(
            "nuscenes-devkit is required for night scene filtering. "
            "Install it with: pip install nuscenes-devkit"
        )

    night_sample_tokens = set()
    for txt_path in [trainval_txt, test_txt]:
        txt_path = Path(txt_path)
        if not txt_path.exists():
            print(f"  Warning: night scenes file not found: {txt_path} – skipping")
            continue
        with open(txt_path, 'r') as f:
            tokens = {line.strip() for line in f if line.strip()}
        print(f"  Loaded {len(tokens)} tokens from {txt_path.name}")
        night_sample_tokens |= tokens

    print(f"  Total merged night sample tokens: {len(night_sample_tokens)}")

    if not night_sample_tokens:
        return set()

    night_filenames = set()
    for version in ('v1.0-trainval', 'v1.0-test'):
        try:
            nusc = NuScenes(version=version, dataroot=nuscenes_dataroot, verbose=False)
        except Exception as e:
            print(f"  Warning: could not load NuScenes {version}: {e} – skipping")
            continue

        for sd in nusc.sample_data:
            if sd['sensor_modality'] != 'camera':
                continue
            if sd['sample_token'] in night_sample_tokens:
                night_filenames.add(Path(sd['filename']).name)

    print(f"  Total night image filenames resolved: {len(night_filenames)}")
    return night_filenames


class ImageQualityAnalyzer:
    def __init__(self, input_directory, output_directory, gpu_id=0, batch_size=16,
                 num_workers=None, sample_size=1000,
                 night_trainval_txt=None, night_test_txt=None, nuscenes_dataroot=None,
                 image_size=(256, 256)):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.results = {}
        self.gpu_id = gpu_id
        self.use_gpu = False
        self.batch_size = batch_size
        self.num_workers = num_workers or min(cpu_count(), 8)
        self.sample_size = sample_size
        self.image_size = image_size  # (width, height) or None to disable resizing

        # Night-scene filtering
        self.night_filenames = set()
        if night_trainval_txt and night_test_txt and nuscenes_dataroot:
            print("Loading night scene filter…")
            self.night_filenames = load_night_scene_filenames(
                night_trainval_txt, night_test_txt, nuscenes_dataroot
            )
            print(f"Night scene filter ready: {len(self.night_filenames)} images will be excluded.")
        else:
            print("No night scene filter configured – all images will be processed.")

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

    def resize_image(self, image):
        """Resize image to target size before analysis.

        Skips resize if image_size is None or image is already the correct size.
        Uses INTER_AREA interpolation which is optimal for downscaling.
        """
        if self.image_size is None:
            return image
        target_w, target_h = self.image_size
        h, w = image.shape[:2]
        if w == target_w and h == target_h:
            return image
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def initialize_gpu(self):
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
                if self.gpu_id < cv2.cuda.getCudaEnabledDeviceCount():
                    cv2.cuda.setDevice(self.gpu_id)
                    self.use_gpu = True
                    print(f"Using GPU {self.gpu_id} for processing")
                    print(f"Batch size: {self.batch_size}")
                    print(f"Worker threads: {self.num_workers}")
                    print(f"Sample size per camera: {self.sample_size} images")
                    if self.image_size:
                        print(f"Images resized to: {self.image_size[0]}x{self.image_size[1]} before analysis")
                    else:
                        print("Image resizing disabled (original resolution)")
                    cv2.cuda.printCudaDeviceInfo(self.gpu_id)
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
        if not self.use_gpu:
            return
        try:
            if self.image_size:
                w, h = self.image_size
                common_sizes = [(h, w)]
            else:
                common_sizes = [(1920, 1080), (1280, 720), (640, 480), (2048, 1024)]

            for size in common_sizes:
                for _ in range(2):
                    gpu_mat = cv2.cuda_GpuMat(*size, cv2.CV_8UC3)
                    self.gpu_memory_pool.append(gpu_mat)

            print(f"Pre-allocated {len(self.gpu_memory_pool)} GPU memory buffers")
        except Exception as e:
            print(f"GPU memory pool initialization failed: {e}")
            self.gpu_memory_pool = []

    def get_gpu_buffer(self, rows, cols, dtype=cv2.CV_8UC3):
        if not self.use_gpu:
            return None
        with self.gpu_lock:
            for i, buffer in enumerate(self.gpu_memory_pool):
                if buffer.rows >= rows and buffer.cols >= cols:
                    return self.gpu_memory_pool.pop(i)
        try:
            return cv2.cuda_GpuMat(rows, cols, dtype)
        except Exception:
            return None

    def return_gpu_buffer(self, buffer):
        if not self.use_gpu or buffer is None:
            return
        with self.gpu_lock:
            if len(self.gpu_memory_pool) < 20:
                self.gpu_memory_pool.append(buffer)

    def process_batch_gpu(self, image_batch):
        if not self.use_gpu or not image_batch:
            return []

        try:
            batch_results = []
            gpu_images = []
            gpu_grays = []

            for image_data in image_batch:
                image, filename = image_data
                gpu_buffer = self.get_gpu_buffer(image.shape[0], image.shape[1], cv2.CV_8UC3)

                if gpu_buffer is not None:
                    gpu_buffer.upload(image)
                    gpu_images.append((gpu_buffer, filename))
                    gpu_gray = cv2.cuda.cvtColor(gpu_buffer, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else gpu_buffer
                    gpu_grays.append(gpu_gray)
                else:
                    gpu_images.append((image, filename))
                    gpu_grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image)

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

            for stream in streams:
                stream.waitForCompletion()

            for gpu_img, _ in gpu_images:
                if isinstance(gpu_img, cv2.cuda_GpuMat):
                    self.return_gpu_buffer(gpu_img)

            return batch_results

        except Exception as e:
            print(f"Batch processing failed: {e}")
            return []

    def calculate_metrics_gpu_parallel(self, gpu_image, gpu_gray, stream):
        metrics = {}
        try:
            brightness_future = self.calculate_brightness_gpu_async(gpu_gray, stream)
            sharpness_future = self.calculate_sharpness_gpu_async(gpu_gray, stream)
            artifacts_future = self.calculate_compression_artifacts_gpu_async(gpu_gray, stream)
            noise_future = self.calculate_noise_gpu_async(gpu_gray, stream)
            chromatic_aberration_future = self.calculate_chromatic_aberration_gpu_async(gpu_image, stream)

            # Download once for all CPU-side metrics
            image_cpu = gpu_image.download() if isinstance(gpu_image, cv2.cuda_GpuMat) else gpu_image
            gray_cpu = gpu_gray.download() if isinstance(gpu_gray, cv2.cuda_GpuMat) else gpu_gray

            metrics['vignetting'] = self.calculate_vignetting_cpu(gray_cpu)
            metrics['saturation'] = self.calculate_saturation_cpu(image_cpu)
            metrics['laplacian_variance'] = self.calculate_laplacian_variance_cpu(gray_cpu)
            metrics['fft_power_spectrum'] = self.calculate_fft_power_spectrum_cpu(gray_cpu)

            stream.waitForCompletion()

            metrics['brightness'] = brightness_future
            metrics['sharpness'] = sharpness_future
            metrics['compression_artifacts'] = artifacts_future
            metrics['noise'] = noise_future
            metrics['chromatic_aberration'] = chromatic_aberration_future

            return metrics

        except Exception as e:
            print(f"GPU parallel metrics calculation failed: {e}")
            return self.calculate_metrics_cpu_fallback(
                gpu_image.download() if isinstance(gpu_image, cv2.cuda_GpuMat) else gpu_image
            )

    def calculate_brightness_gpu_async(self, gpu_gray, stream):
        try:
            sum_result = cv2.cuda.sum(gpu_gray)
            return float(sum_result[0] / (gpu_gray.rows * gpu_gray.cols))
        except:
            return 0.0

    def calculate_sharpness_gpu_async(self, gpu_gray, stream):
        try:
            gpu_laplacian = cv2.cuda.Laplacian(gpu_gray, cv2.CV_64F)
            laplacian_cpu = gpu_laplacian.download()
            return float(np.var(laplacian_cpu))
        except:
            return 0.0

    def calculate_compression_artifacts_gpu_async(self, gpu_gray, stream):
        try:
            gpu_grad_x = cv2.cuda.Sobel(gpu_gray, cv2.CV_64F, 1, 0, ksize=3)
            gpu_grad_y = cv2.cuda.Sobel(gpu_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_x = gpu_grad_x.download()
            grad_y = gpu_grad_y.download()
            return float(np.var(np.sqrt(grad_x**2 + grad_y**2)))
        except:
            return 0.0

    def calculate_noise_gpu_async(self, gpu_gray, stream):
        try:
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
            gray_cpu = gpu_gray.download()
            blurred_cpu = gpu_blurred.download()
            return float(np.std(gray_cpu.astype(np.float64) - blurred_cpu.astype(np.float64)))
        except:
            return 0.0

    def calculate_chromatic_aberration_gpu_async(self, gpu_image, stream):
        try:
            gpu_channels = cv2.cuda.split(gpu_image)
            if len(gpu_channels) >= 3:
                blue_edges = cv2.cuda.Canny(gpu_channels[0], 50, 150).download()
                red_edges  = cv2.cuda.Canny(gpu_channels[2], 50, 150).download()
                return float(np.mean(np.abs(blue_edges.astype(np.float32) - red_edges.astype(np.float32))))
            return 0.0
        except:
            return 0.0

    # ------------------------------------------------------------------ #
    #  New metrics – CPU implementations                                   #
    # ------------------------------------------------------------------ #

    def calculate_saturation_cpu(self, image):
        """Mean saturation of the S-channel in HSV space.
        Synthetic images tend to be oversaturated vs. real nuScenes images."""
        if len(image.shape) < 3 or image.shape[2] < 3:
            return 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))

    def calculate_laplacian_variance_cpu(self, gray_image):
        """Laplacian variance with ksize=5 for mid-frequency sharpness sensitivity.
        Complements the existing 'sharpness' metric (ksize=1) — synthetic images
        often appear unnaturally sharp at mid frequencies."""
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray_image.astype(np.float64), cv2.CV_64F, ksize=5)
        return float(np.var(lap))

    def calculate_fft_power_spectrum_cpu(self, gray_image):
        """High-frequency energy ratio from the 2-D FFT magnitude spectrum.
        Returns the fraction of total power in the outer 50% of frequencies.
        Real camera images have richer high-freq content from grain and lens
        imperfections than synthetic renders."""
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray_image.astype(np.float64))
        magnitude = np.abs(np.fft.fftshift(f))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_radius = np.sqrt(cx**2 + cy**2)

        total_power = np.sum(magnitude)
        high_freq_power = np.sum(magnitude[dist > 0.5 * max_radius])
        return float(high_freq_power / total_power) if total_power > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  CPU fallback (all metrics)                                          #
    # ------------------------------------------------------------------ #

    def calculate_metrics_cpu_fallback(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return {
            'brightness': self.calculate_brightness_cpu(image),
            'sharpness': self.calculate_sharpness_cpu(image),
            'vignetting': self.calculate_vignetting_cpu(gray),
            'compression_artifacts': self.calculate_compression_artifacts_cpu(image),
            'noise': self.calculate_noise_cpu(image),
            'chromatic_aberration': self.calculate_chromatic_aberration_cpu(image),
            'saturation': self.calculate_saturation_cpu(image),
            'laplacian_variance': self.calculate_laplacian_variance_cpu(gray),
            'fft_power_spectrum': self.calculate_fft_power_spectrum_cpu(gray),
        }

    def calculate_brightness_cpu(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.mean(gray))

    def calculate_sharpness_cpu(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def calculate_vignetting_cpu(self, gray_image):
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY) if len(gray_image.shape) == 3 else gray_image
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        center_size = min(h, w) // 10
        center_brightness = np.mean(gray[center_h - center_size:center_h + center_size,
                                         center_w - center_size:center_w + center_size])
        corner_size = min(h, w) // 20
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:]
        ]
        corner_brightness = np.mean([np.mean(c) for c in corners])
        return float((center_brightness - corner_brightness) / center_brightness) if center_brightness > 0 else 0.0

    def calculate_compression_artifacts_cpu(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.var(np.sqrt(grad_x**2 + grad_y**2)))

    def calculate_noise_cpu(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return float(np.std(gray.astype(np.float64) - blurred.astype(np.float64)))

    def calculate_chromatic_aberration_cpu(self, image):
        if len(image.shape) < 3:
            return 0.0
        blue_edges = cv2.Canny(image[:, :, 0], 50, 150)
        red_edges  = cv2.Canny(image[:, :, 2], 50, 150)
        return float(np.mean(np.abs(blue_edges.astype(np.float32) - red_edges.astype(np.float32))))

    # ------------------------------------------------------------------ #
    #  Image loading / batch processing                                    #
    # ------------------------------------------------------------------ #

    def analyze_image_batch(self, image_paths_batch):
        if self.use_gpu:
            image_batch = []
            for image_path in image_paths_batch:
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        image = self.resize_image(image)
                        image_batch.append((image, image_path.name))
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    continue
            if image_batch:
                return self.process_batch_gpu(image_batch)
            return []
        else:
            return self.process_batch_cpu_parallel(image_paths_batch)

    def process_batch_cpu_parallel(self, image_paths_batch):
        results = []

        def process_single_image(image_path):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    return None
                image = self.resize_image(image)
                metrics = self.calculate_metrics_cpu_fallback(image)
                metrics['filename'] = image_path.name
                return metrics
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(process_single_image, path): path for path in image_paths_batch}
            for future in future_to_path:
                result = future.result()
                if result is not None:
                    results.append(result)

        return results

    def process_directory(self, directory_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []

        all_image_files = [f for f in directory_path.glob('*') if f.suffix.lower() in image_extensions]

        # Night-scene filtering
        if self.night_filenames:
            before = len(all_image_files)
            all_image_files = [f for f in all_image_files if f.name not in self.night_filenames]
            filtered = before - len(all_image_files)
            if filtered:
                print(f"  Night filter removed {filtered} images from {directory_path.name} "
                      f"({before} → {len(all_image_files)})")

        total_available = len(all_image_files)

        if total_available == 0:
            print(f"  No images found in {directory_path.name}")
            return results

        if total_available <= self.sample_size:
            print(f"  Found {total_available} images (using all available)")
            image_files = all_image_files
        else:
            print(f"  Found {total_available} images (randomly sampling {self.sample_size})")
            random.seed(42)
            image_files = random.sample(all_image_files, self.sample_size)

        total_images = len(image_files)
        if self.image_size:
            print(f"  Processing {total_images} images (resized to {self.image_size[0]}x{self.image_size[1]})...")
        else:
            print(f"  Processing {total_images} images (original resolution)...")

        start_time = time.time()
        processed_count = 0

        for i in range(0, total_images, self.batch_size):
            batch = image_files[i:i + self.batch_size]
            batch_start_time = time.time()

            batch_results = self.analyze_image_batch(batch)
            results.extend(batch_results)

            processed_count += len(batch)
            batch_time = time.time() - batch_start_time

            if self.use_gpu:
                throughput = len(batch) / batch_time if batch_time > 0 else 0
                print(f"  Batch {i // self.batch_size + 1}: {len(batch)} images in {batch_time:.2f}s "
                      f"({throughput:.1f} img/s) - Total: {processed_count}/{total_images}")
            else:
                print(f"  Progress: {processed_count}/{total_images} images processed")

        total_time = time.time() - start_time
        print(f"  Directory completed: {total_images} images in {total_time:.2f}s "
              f"({total_images / total_time:.1f} img/s)")

        return results

    # ------------------------------------------------------------------ #
    #  Statistics & plotting                                               #
    # ------------------------------------------------------------------ #

    def calculate_statistics(self, data, metrics):
        stats = {}
        for metric in metrics:
            values = [item[metric] for item in data if metric in item]
            stats[metric] = {
                'average': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'values': values
            }
        return stats

    def create_histogram_plots(self, stats, title, save_path):
        """Grid of histogram plots for all metrics (3 per row)."""
        n = len(ALL_METRICS)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, metric in enumerate(ALL_METRICS):
            ax = axes[i]
            if metric not in stats:
                ax.set_visible(False)
                continue
            values  = stats[metric]['values']
            avg     = stats[metric]['average']
            median  = stats[metric]['median']
            std_val = stats[metric]['std']

            ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(avg,    color='red',   linestyle='--', linewidth=2, label=f'Avg: {avg:.3f}')
            ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.3f}')
            ax.set_title(f'{METRIC_TITLES[metric]}\n(σ: {std_val:.3f})', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_to_csv(self, stats, filename):
        data = [
            {'metric': m, 'average': stats[m]['average'],
             'median': stats[m]['median'], 'std': stats[m]['std']}
            for m in ALL_METRICS if m in stats
        ]
        pd.DataFrame(data).to_csv(filename, index=False)

    def save_detailed_csv(self, data, filename):
        pd.DataFrame(data).to_csv(filename, index=False)

    def save_histogram_data_to_csv(self, stats, filename, num_bins=20):
        data = []

        for metric in ALL_METRICS:
            if metric not in stats:
                continue
            values        = stats[metric]['values']
            avg           = stats[metric]['average']
            median        = stats[metric]['median']
            std_val       = stats[metric]['std']
            total_samples = len(values)

            hist, bin_edges = np.histogram(values, bins=num_bins)
            bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2
            cumulative_freq = np.cumsum(hist)

            for i in range(num_bins):
                bin_start = bin_edges[i]
                bin_end   = bin_edges[i + 1]
                frequency = hist[i]
                data.append({
                    'metric': metric, 'average': avg, 'median': median, 'std': std_val,
                    'total_samples': total_samples,
                    'min_value': np.min(values), 'max_value': np.max(values),
                    'num_bins': num_bins, 'bin_number': i + 1,
                    'bin_center': bin_centers[i], 'bin_start': bin_start, 'bin_end': bin_end,
                    'frequency': frequency,
                    'frequency_density': frequency / (bin_end - bin_start) if (bin_end - bin_start) > 0 else 0,
                    'cumulative_frequency': cumulative_freq[i],
                    'relative_frequency': frequency / total_samples if total_samples > 0 else 0,
                })

        pd.DataFrame(data).to_csv(filename, index=False)

    def create_combined_metric_plots(self, all_data, metrics):
        cameras = sorted(set(item['camera'] for item in all_data))
        colors  = plt.cm.Set3(np.linspace(0, 1, len(cameras)))

        for metric in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            for i, camera in enumerate(cameras):
                vals = [item[metric] for item in all_data if item['camera'] == camera and metric in item]
                ax.hist(vals, bins=15, alpha=0.6, label=camera, color=colors[i])

            ax.set_title(
                f'Distribution of {METRIC_TITLES.get(metric, metric)} Across All Cameras',
                fontweight='bold', fontsize=14
            )
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_directory / f"combined_{metric}_distribution.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            metric_data = [
                {
                    'camera': cam,
                    'average': np.mean([item[metric] for item in all_data if item['camera'] == cam and metric in item]),
                    'median':  np.median([item[metric] for item in all_data if item['camera'] == cam and metric in item]),
                    'std':     np.std([item[metric] for item in all_data if item['camera'] == cam and metric in item]),
                }
                for cam in cameras
            ]
            pd.DataFrame(metric_data).to_csv(
                self.output_directory / f"combined_{metric}_statistics.csv", index=False
            )

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run_analysis(self):
        all_camera_data = []

        print(f"Input directory: {self.input_directory}")
        print(f"Output directory: {self.output_directory}")
        print(f"Sample size per camera: {self.sample_size} images")
        if self.image_size:
            print(f"Analysis image size: {self.image_size[0]}x{self.image_size[1]} (resized if needed)")
        else:
            print("Analysis image size: original resolution (no resizing)")
        print(f"Metrics: {', '.join(ALL_METRICS)}")
        print("=" * 50)

        for camera_dir in sorted(self.input_directory.iterdir()):
            if not (camera_dir.is_dir() and camera_dir.name.startswith('CAM_')):
                continue

            print(f"Processing {camera_dir.name}...")
            image_data = self.process_directory(camera_dir)
            if not image_data:
                print(f"No valid images found in {camera_dir.name}")
                continue

            stats = self.calculate_statistics(image_data, ALL_METRICS)
            self.results[camera_dir.name] = stats

            self.create_histogram_plots(
                stats, f"Image Quality Analysis – {camera_dir.name}",
                self.output_directory / f"{camera_dir.name}_analysis.png"
            )
            self.save_to_csv(stats, self.output_directory / f"{camera_dir.name}_statistics.csv")
            self.save_detailed_csv(image_data, self.output_directory / f"{camera_dir.name}_detailed.csv")

            for item in image_data:
                item['camera'] = camera_dir.name
                all_camera_data.append(item)

        if not all_camera_data:
            print("No data found. Please check your directory structure.")
            print("Expected structure: input_dir/CAM_*/image_files.jpg")
            return

        print("Creating combined analysis...")
        self.create_combined_metric_plots(all_camera_data, ALL_METRICS)

        print("Creating overall analysis...")
        overall_stats = self.calculate_statistics(all_camera_data, ALL_METRICS)

        self.create_histogram_plots(
            overall_stats, "Overall Image Quality Analysis – All nuScenes Cameras",
            self.output_directory / "overall_analysis.png"
        )
        self.save_histogram_data_to_csv(overall_stats, self.output_directory / "overall_statistics.csv")
        self.save_detailed_csv(all_camera_data, self.output_directory / "all_cameras_detailed.csv")

        print("=" * 50)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_directory}")

        print("\nOverall Summary:")
        for metric in ALL_METRICS:
            print(f"  {METRIC_TITLES[metric]}:")
            print(f"    Average: {overall_stats[metric]['average']:.4f}")
            print(f"    Median:  {overall_stats[metric]['median']:.4f}")
            print(f"    Std Dev: {overall_stats[metric]['std']:.4f}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze image quality metrics for camera directories with GPU acceleration'
    )
    parser.add_argument('input_dir',  help='Input directory containing CAM_* subdirectories')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU processing')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for GPU (default: 16)')
    parser.add_argument('--workers', type=int, default=None, help='CPU worker threads (default: auto)')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Images sampled per camera directory (default: 1000)')
    parser.add_argument('--night-trainval-txt', type=str, default=None,
                        help='Path to the nuScenes v1.0-trainval night scenes txt file')
    parser.add_argument('--night-test-txt', type=str, default=None,
                        help='Path to the nuScenes v1.0-test night scenes txt file')
    parser.add_argument('--nuscenes-dataroot', type=str, default=None,
                        help='Root directory of the nuScenes dataset (required for night filtering)')
    parser.add_argument('--image-size', type=str, default='256x256',
                        help='Resize before analysis, e.g. 640x480. Use "none" to disable (default: 256x256).')
    return parser.parse_args()


def validate_directories(input_dir, output_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return False
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory!")
        return False
    cam_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('CAM_')]
    if not cam_dirs:
        print(f"Warning: No CAM_* directories found in '{input_dir}'")
        print("Expected directory structure: input_dir/CAM_*/image_files.jpg")
    return True


def main():
    args = parse_arguments()

    if not validate_directories(args.input_dir, args.output_dir):
        sys.exit(1)

    if args.image_size.lower() == 'none':
        image_size = None
    else:
        try:
            w, h = map(int, args.image_size.lower().split('x'))
            if w <= 0 or h <= 0:
                raise ValueError
            image_size = (w, h)
        except (ValueError, AttributeError):
            print(f"Error: Invalid image size '{args.image_size}'. Use WxH (e.g. 256x256) or 'none'.")
            sys.exit(1)

    print("Image Quality Analysis Tool with GPU Acceleration")
    print("=" * 80)
    if image_size:
        print(f"Image resize: {image_size[0]}x{image_size[1]} (skipped if already correct size)")
    else:
        print("Image resize: disabled (original resolution)")

    night_trainval_txt = args.night_trainval_txt
    night_test_txt     = args.night_test_txt
    nuscenes_dataroot  = args.nuscenes_dataroot

    if (night_trainval_txt or night_test_txt) and not nuscenes_dataroot:
        print("Warning: --night-trainval-txt / --night-test-txt provided but "
              "--nuscenes-dataroot is missing. Night filtering will be skipped.")
        night_trainval_txt = night_test_txt = None

    common_kwargs = dict(
        batch_size=args.batch_size if not args.no_gpu else 1,
        num_workers=args.workers,
        sample_size=args.sample_size,
        night_trainval_txt=night_trainval_txt,
        night_test_txt=night_test_txt,
        nuscenes_dataroot=nuscenes_dataroot,
        image_size=image_size,
    )

    if args.no_gpu:
        print("GPU processing disabled by user")
        analyzer = ImageQualityAnalyzer(args.input_dir, args.output_dir,
                                        gpu_id=-1, **common_kwargs)
        analyzer.use_gpu = False
    else:
        analyzer = ImageQualityAnalyzer(args.input_dir, args.output_dir,
                                        gpu_id=args.gpu, **common_kwargs)

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