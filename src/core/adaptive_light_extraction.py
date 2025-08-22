#!/usr/bin/env python3
"""
Adaptive Light Intensity Pixel Extraction

This script provides multiple adaptive methods for extracting light intensity pixels
from images, handling various lighting conditions and image characteristics.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import argparse
import os
from typing import Tuple, Optional, Dict, Any


class AdaptiveLightExtractor:
    """
    A comprehensive class for adaptive light intensity pixel extraction
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the extractor with an image
        
        Args:
            image_path: Path to the input image
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to different color spaces for analysis
        self.bgr_image = self.original_image.copy()
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.lab_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        
        self.height, self.width = self.gray_image.shape
        
    def analyze_image_statistics(self) -> Dict[str, Any]:
        """
        Analyze image statistics to guide adaptive threshold selection
        
        Returns:
            Dictionary containing various image statistics
        """
        stats = {
            'mean_intensity': np.mean(self.gray_image),
            'std_intensity': np.std(self.gray_image),
            'min_intensity': np.min(self.gray_image),
            'max_intensity': np.max(self.gray_image),
            'median_intensity': np.median(self.gray_image),
            'contrast': np.std(self.gray_image) / np.mean(self.gray_image) if np.mean(self.gray_image) > 0 else 0,
            'dynamic_range': np.max(self.gray_image) - np.min(self.gray_image)
        }
        
        # Analyze histogram
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        stats['histogram_peak'] = np.argmax(hist)
        stats['histogram_skewness'] = self._calculate_skewness(hist.flatten())
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def adaptive_threshold_otsu(self) -> np.ndarray:
        """
        Extract light pixels using Otsu's adaptive thresholding
        
        Returns:
            Binary mask of light intensity pixels
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        
        # Otsu's thresholding
        threshold_value, binary_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        print(f"Otsu threshold value: {threshold_value}")
        return binary_mask
    
    def adaptive_threshold_local(self, block_size: int = 15, C: int = 2) -> np.ndarray:
        """
        Extract light pixels using local adaptive thresholding
        
        Args:
            block_size: Size of the neighborhood for threshold calculation
            C: Constant subtracted from the mean
            
        Returns:
            Binary mask of light intensity pixels
        """
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        
        # Adaptive threshold
        binary_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        
        return binary_mask
    
    def adaptive_threshold_percentile(self, percentile: float = 85) -> np.ndarray:
        """
        Extract light pixels using percentile-based adaptive thresholding
        
        Args:
            percentile: Percentile value for threshold (0-100)
            
        Returns:
            Binary mask of light intensity pixels
        """
        threshold_value = np.percentile(self.gray_image, percentile)
        binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
        
        print(f"Percentile ({percentile}%) threshold value: {threshold_value}")
        return binary_mask
    
    def adaptive_threshold_kmeans(self, n_clusters: int = 3) -> np.ndarray:
        """
        Extract light pixels using K-means clustering
        
        Args:
            n_clusters: Number of clusters for K-means
            
        Returns:
            Binary mask of light intensity pixels
        """
        # Reshape image for K-means
        pixel_values = self.gray_image.reshape((-1, 1)).astype(np.float32)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Find the cluster with highest intensity (brightest)
        centers = centers.flatten()
        brightest_cluster = np.argmax(centers)
        
        # Create mask for the brightest cluster
        labels = labels.reshape(self.gray_image.shape)
        binary_mask = (labels == brightest_cluster).astype(np.uint8) * 255
        
        print(f"K-means cluster centers: {centers}")
        print(f"Brightest cluster: {brightest_cluster} with intensity: {centers[brightest_cluster]}")
        
        return binary_mask
    
    def adaptive_threshold_hsv(self, value_threshold: float = 0.7, saturation_max: float = 0.3) -> np.ndarray:
        """
        Extract light pixels using HSV color space analysis
        
        Args:
            value_threshold: Minimum value (brightness) threshold (0-1)
            saturation_max: Maximum saturation threshold (0-1)
            
        Returns:
            Binary mask of light intensity pixels
        """
        # Normalize HSV to 0-1 range
        hsv_normalized = self.hsv_image.astype(np.float32) / 255.0
        
        # Extract high value (brightness) and low saturation pixels
        value_mask = hsv_normalized[:, :, 2] >= value_threshold
        saturation_mask = hsv_normalized[:, :, 1] <= saturation_max
        
        # Combine masks
        combined_mask = value_mask & saturation_mask
        binary_mask = combined_mask.astype(np.uint8) * 255
        
        return binary_mask
    
    def adaptive_threshold_lab(self, lightness_threshold: float = 0.7) -> np.ndarray:
        """
        Extract light pixels using LAB color space (L channel)
        
        Args:
            lightness_threshold: Minimum lightness threshold (0-1)
            
        Returns:
            Binary mask of light intensity pixels
        """
        # Normalize L channel to 0-1 range
        l_channel = self.lab_image[:, :, 0].astype(np.float32) / 255.0
        
        # Apply threshold to L channel
        binary_mask = (l_channel >= lightness_threshold).astype(np.uint8) * 255
        
        return binary_mask
    
    def adaptive_threshold_statistical(self, std_multiplier: float = 1.5) -> np.ndarray:
        """
        Extract light pixels using statistical analysis
        
        Args:
            std_multiplier: Multiplier for standard deviation
            
        Returns:
            Binary mask of light intensity pixels
        """
        stats = self.analyze_image_statistics()
        
        # Calculate adaptive threshold based on statistics
        threshold_value = stats['mean_intensity'] + (std_multiplier * stats['std_intensity'])
        threshold_value = min(threshold_value, 255)  # Clamp to valid range
        
        binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
        
        print(f"Statistical threshold value: {threshold_value}")
        print(f"Based on mean: {stats['mean_intensity']:.2f}, std: {stats['std_intensity']:.2f}")
        
        return binary_mask
    
    def morphological_cleanup(self, binary_mask: np.ndarray, 
                            kernel_size: int = 3, 
                            operations: str = "close") -> np.ndarray:
        """
        Apply morphological operations to clean up the binary mask
        
        Args:
            binary_mask: Input binary mask
            kernel_size: Size of morphological kernel
            operations: Type of operations ("open", "close", "open_close")
            
        Returns:
            Cleaned binary mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operations == "open":
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        elif operations == "close":
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        elif operations == "open_close":
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        else:
            cleaned = binary_mask
            
        return cleaned
    
    def extract_connected_components(self, binary_mask: np.ndarray, 
                                   min_area: int = 50) -> Tuple[np.ndarray, Dict]:
        """
        Extract connected components and filter by area
        
        Args:
            binary_mask: Input binary mask
            min_area: Minimum area threshold for components
            
        Returns:
            Tuple of (filtered mask, component statistics)
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Filter components by area
        filtered_mask = np.zeros_like(binary_mask)
        valid_components = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                component_mask = (labels == i)
                filtered_mask[component_mask] = 255
                valid_components.append({
                    'label': i,
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                })
        
        component_info = {
            'total_components': len(valid_components),
            'components': valid_components
        }
        
        return filtered_mask, component_info
    
    def auto_select_method(self) -> str:
        """
        Automatically select the best extraction method based on image characteristics
        
        Returns:
            Recommended method name
        """
        stats = self.analyze_image_statistics()
        
        # Decision logic based on image characteristics
        if stats['contrast'] < 0.3:  # Low contrast images
            return "hsv"
        elif stats['dynamic_range'] < 100:  # Low dynamic range
            return "statistical"
        elif stats['histogram_skewness'] > 1:  # Right-skewed (many dark pixels)
            return "percentile"
        elif stats['std_intensity'] > 50:  # High variation
            return "local"
        else:  # Default case
            return "otsu"
    
    def extract_all_methods(self) -> Dict[str, np.ndarray]:
        """
        Extract light pixels using all available methods
        
        Returns:
            Dictionary mapping method names to binary masks
        """
        methods = {
            'otsu': self.adaptive_threshold_otsu,
            'local': self.adaptive_threshold_local,
            'percentile': self.adaptive_threshold_percentile,
            'kmeans': self.adaptive_threshold_kmeans,
            'hsv': self.adaptive_threshold_hsv,
            'lab': self.adaptive_threshold_lab,
            'statistical': self.adaptive_threshold_statistical
        }
        
        results = {}
        for name, method in methods.items():
            try:
                print(f"\nExtracting using {name} method...")
                mask = method()
                # Apply morphological cleanup
                cleaned_mask = self.morphological_cleanup(mask)
                results[name] = cleaned_mask
            except Exception as e:
                print(f"Error in {name} method: {e}")
                results[name] = np.zeros_like(self.gray_image)
        
        return results
    
    def visualize_results(self, results: Dict[str, np.ndarray], 
                         save_path: Optional[str] = None):
        """
        Visualize extraction results for all methods
        
        Args:
            results: Dictionary of method results
            save_path: Optional path to save the visualization
        """
        n_methods = len(results)
        n_cols = 3
        n_rows = (n_methods + 2) // n_cols  # +1 for original image
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Show original image
        axes[0].imshow(self.rgb_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show results for each method
        for i, (method_name, mask) in enumerate(results.items(), 1):
            if i < len(axes):
                axes[i].imshow(mask, cmap='gray')
                axes[i].set_title(f'{method_name.upper()} Method')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate a comprehensive report of extraction results
        
        Args:
            results: Dictionary of method results
            
        Returns:
            Report dictionary with statistics for each method
        """
        stats = self.analyze_image_statistics()
        report = {
            'image_info': {
                'path': self.image_path,
                'dimensions': (self.width, self.height),
                'total_pixels': self.width * self.height
            },
            'image_statistics': stats,
            'method_results': {}
        }
        
        for method_name, mask in results.items():
            # Count light pixels
            light_pixels = np.sum(mask > 0)
            light_percentage = (light_pixels / (self.width * self.height)) * 100
            
            # Get connected components info
            filtered_mask, component_info = self.extract_connected_components(mask)
            
            report['method_results'][method_name] = {
                'light_pixels_count': int(light_pixels),
                'light_pixels_percentage': round(light_percentage, 2),
                'connected_components': component_info['total_components'],
                'largest_component_area': max([c['area'] for c in component_info['components']], default=0)
            }
        
        return report


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Adaptive Light Intensity Pixel Extraction')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--method', choices=['auto', 'otsu', 'local', 'percentile', 'kmeans', 
                                           'hsv', 'lab', 'statistical', 'all'], 
                       default='auto', help='Extraction method to use')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    # Initialize extractor
    try:
        extractor = AdaptiveLightExtractor(args.image_path)
        print(f"Successfully loaded image: {args.image_path}")
        print(f"Image dimensions: {extractor.width} x {extractor.height}")
        
        # Analyze image statistics
        stats = extractor.analyze_image_statistics()
        print(f"\nImage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Extract using specified method
        if args.method == 'auto':
            recommended_method = extractor.auto_select_method()
            print(f"\nAuto-selected method: {recommended_method}")
            method_func = getattr(extractor, f'adaptive_threshold_{recommended_method}')
            mask = method_func()
            results = {recommended_method: extractor.morphological_cleanup(mask)}
        elif args.method == 'all':
            print(f"\nExtracting using all methods...")
            results = extractor.extract_all_methods()
        else:
            print(f"\nExtracting using {args.method} method...")
            method_func = getattr(extractor, f'adaptive_threshold_{args.method}')
            mask = method_func()
            results = {args.method: extractor.morphological_cleanup(mask)}
        
        # Generate report
        if args.report:
            report = extractor.generate_report(results)
            print(f"\n{'='*50}")
            print("EXTRACTION REPORT")
            print(f"{'='*50}")
            print(f"Image: {report['image_info']['path']}")
            print(f"Dimensions: {report['image_info']['dimensions']}")
            print(f"Total pixels: {report['image_info']['total_pixels']:,}")
            print(f"\nMethod Results:")
            for method, result in report['method_results'].items():
                print(f"  {method.upper()}:")
                print(f"    Light pixels: {result['light_pixels_count']:,} ({result['light_pixels_percentage']:.1f}%)")
                print(f"    Components: {result['connected_components']}")
                print(f"    Largest component: {result['largest_component_area']} pixels")
        
        # Save outputs
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            for method_name, mask in results.items():
                output_path = os.path.join(args.output, f"{base_name}_{method_name}_mask.png")
                cv2.imwrite(output_path, mask)
                print(f"Saved {method_name} mask to: {output_path}")
        
        # Visualize
        if args.visualize:
            vis_path = None
            if args.output:
                vis_path = os.path.join(args.output, f"{base_name}_comparison.png")
            extractor.visualize_results(results, vis_path)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
