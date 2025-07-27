"""
Geospatial Sampling and Clipping Module

This module provides comprehensive functionality for geospatial data sampling and clipping operations,
with support for both raster and vector data processing. It includes coordinate generation strategies,
batch processing capabilities, and cross-platform multiprocessing support.

Classes:
    SamplingWindowGenerator: Generates sampling window coordinates using various strategies
        - Centroid-based sampling from vector features
        - Sliding window sampling across raster extent
        - UUID-based feature selection and filtering
        - Support for custom naming fields and coordinate validation
    
    BaseSampler: Base class for single data clipping operations
        - Raster clipping with multiple output formats (array, geocoded)
        - Vector rasterization and clipping with attribute filtering
        - CRS validation and coordinate bounds checking
        - Flexible file saving with customizable naming conventions
    
    Sampler: Advanced batch processing class (inherits from BaseSampler)
        - Parallel processing with automatic worker optimization
        - Progress tracking and error handling
        - Cross-platform multiprocessing (Windows, Linux, macOS)
        - Automatic fallback to serial processing on errors

Key Features:
    • Cross-Platform Compatibility: Automatic OS detection and multiprocessing configuration
    • Robust Error Handling: Comprehensive validation and graceful error recovery
    • Memory Management: Proper GDAL resource cleanup and context management
    • ID Validation: Automatic filtering of null/None/empty ID values
    • Progress Tracking: Real-time progress bars for batch operations
    • Flexible Output: Support for array and geocoded (TIF) output formats
    • CRS Validation: Automatic coordinate reference system checking
    • Resource Optimization: Intelligent worker count management based on CPU cores

Supported Operations:
    Raster Operations:
        - Single and batch raster clipping
        - Multiple resampling algorithms (bilinear, nearest, cubic, etc.)
        - Compression and tiling options for output files
        - Memory-efficient processing for large datasets
    
    Vector Operations:
        - Vector to raster conversion (rasterization)
        - Spatial and attribute filtering
        - Custom burn values and pixel sizes
        - Feature geometry validation and processing

Coordinate Generation Strategies:
    • Centroid-based: Generate windows centered on vector feature centroids
    • UUID-based: Select specific features using unique identifier columns
    • Sliding window: Regular grid sampling across raster extent with configurable stride
    • Custom boundaries: User-defined coordinate sets for targeted sampling

Multiprocessing Support:
    • Windows: Uses 'spawn' method for stability and compatibility
    • Linux/macOS: Uses 'fork' method for efficiency, with 'spawn' fallback
    • Automatic worker validation and CPU core detection
    • Process-safe GDAL initialization in worker processes
    • Error collection and reporting across all workers

File Management:
    • Automatic filename sanitization for cross-platform compatibility
    • Customizable prefixes and naming conventions
    • Directory creation and validation
    • Comprehensive file path handling

Dependencies:
    - GDAL/OGR: Geospatial data reading, writing, and processing
    - NumPy: Array operations and numerical computations
    - OpenCV (cv2): Image processing and file I/O operations
    - tqdm: Progress bar display and tracking
    - psutil: System resource monitoring and CPU detection
    - multiprocessing: Parallel processing capabilities

Usage Examples:
    Basic raster clipping:
        >>> sampler = Sampler("base_raster.tif")
        >>> result = sampler.clip_raster("input.tif", (minx, maxy, maxx, miny))
    
    Batch vector processing:
        >>> generator = SamplingWindowGenerator("base_raster.tif")
        >>> coords, names = generator.get_centroid_coordinates("vector.shp", 512)
        >>> sampler.clip_vectors("vector.shp", coords, names, "output_dir", n_workers=4)
    
    UUID-based feature selection:
        >>> coords, name = generator.get_centroid_coordinate_by_uuid(
        ...     "vector.shp", 512, "id_column", "specific_uuid"
        ... )

Error Handling:
    The module implements comprehensive error handling with automatic recovery mechanisms:
    - Invalid coordinate validation with warnings
    - CRS mismatch detection and reporting
    - Multiprocessing failure recovery with serial fallback
    - Resource cleanup on exceptions
    - Detailed error reporting for debugging

Performance Considerations:
    • Automatic worker count optimization (60% of CPU cores)
    • Memory-efficient processing using GDAL streaming
    • Batch processing to minimize overhead
    • Context managers for proper resource cleanup
    • Progress tracking without performance impact

Platform Notes:
    Windows: Requires proper multiprocessing protection (if __name__ == '__main__':)
    Linux/macOS: Full multiprocessing support with fork efficiency
    All platforms: Automatic GDAL configuration and error handling

"""

from osgeo import gdal, osr, ogr
import numpy as np
import os
import cv2
import warnings
import multiprocessing as mp
from functools import partial
import psutil
from tqdm import tqdm
import platform
import sys
import tempfile
import uuid
import time
import re

# Global GDAL initialization flag
_gdal_initialized = False

def _setup_multiprocessing():
    """Setup multiprocessing based on the operating system."""
    system = platform.system().lower()
    
    if system == 'windows':
        # Windows-specific multiprocessing setup
        mp.freeze_support()
        
        # Set spawn method explicitly for consistency
        if hasattr(mp, 'set_start_method'):
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # Method already set, ignore
                pass
                
        print("Windows multiprocessing configured with spawn method")
        
    elif system in ['linux', 'darwin']:  # Linux or macOS
        # Unix-like systems can use fork (more efficient)
        if hasattr(mp, 'set_start_method'):
            try:
                # Try fork first (most efficient), fallback to spawn
                mp.set_start_method('fork', force=True)
                print(f"{system.capitalize()} multiprocessing configured with fork method")
            except (RuntimeError, OSError):
                try:
                    mp.set_start_method('spawn', force=True)
                    print(f"{system.capitalize()} multiprocessing configured with spawn method")
                except RuntimeError:
                    pass
    else:
        print(f"Unknown OS ({system}), using default multiprocessing method")

def _initialize_gdal_global():
    """Initialize GDAL globally to prevent segmentation faults."""
    global _gdal_initialized
    if not _gdal_initialized:
        # CRITICAL: Enable exceptions explicitly
        gdal.UseExceptions()
        ogr.UseExceptions()
        osr.UseExceptions()
        
        # Configure GDAL options for stability
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
        gdal.SetConfigOption('CPL_DEBUG', 'OFF')
        gdal.SetConfigOption('GDAL_CACHEMAX', '256')
        gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '100')
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        
        # Push error handler
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='osgeo.gdal')
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        _gdal_initialized = True
        print("GDAL initialized successfully")

# Initialize both GDAL and multiprocessing when module is imported
_initialize_gdal_global()
_setup_multiprocessing()

def _is_valid_id(field_value):
    """Check if an ID field value is valid (not None, null, empty, or whitespace)."""
    if field_value is None:
        return False
    
    str_value = str(field_value).strip()
    if str_value == '' or str_value.lower() in ['none', 'null', 'na', 'n/a']:
        return False
    
    return True

def _sanitize_filename(name):
    """Sanitize filename for valid names only."""
    # Convert to string and sanitize
    name = str(name)
    
    # Replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = name.strip('._')  # Remove leading/trailing dots and underscores
    
    # Limit length
    if len(name) > 100:
        name = name[:100]
    
    return name

def _create_simple_filepath(output_dir, prefix, name, extension=".tif"):
    """Create a simple filepath with format: prefix_name.tif"""
    # Name should already be sanitized and valid at this point
    sanitized_name = _sanitize_filename(name)
    
    filename = f"{prefix}{sanitized_name}{extension}"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return filepath

# Helper functions for multiprocessing
def _init_gdal():
    """Initialize GDAL configuration for worker processes."""
    try:
        # Enable exceptions in child processes
        gdal.UseExceptions()
        ogr.UseExceptions()
        osr.UseExceptions()
        
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
        gdal.SetConfigOption('CPL_DEBUG', 'OFF')
        gdal.SetConfigOption('GDAL_CACHEMAX', '128')
        gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '50')
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
    except Exception as e:
        print(f"Warning: GDAL initialization in worker process failed: {e}")

def _validate_multiprocessing_context():
    """Validate that multiprocessing is properly configured."""
    if platform.system().lower() == 'windows':
        # Check if we're in the main process
        if hasattr(sys, 'frozen') or __name__ != '__main__':
            # We're either in a frozen executable or imported module - this is fine
            return True
        else:
            # We're in a Windows script being run directly
            import inspect
            frame = inspect.currentframe()
            try:
                # Check if we're being called from within an if __name__ == '__main__': block
                while frame:
                    frame = frame.f_back
                    if frame and frame.f_code.co_name == '<module>':
                        # We found the module level - check for proper protection
                        filename = frame.f_code.co_filename
                        if filename.endswith('main.py') or 'main' in filename:
                            print("Warning: On Windows, make sure your main script uses 'if __name__ == \"__main__\":' protection")
                            return False
            finally:
                del frame
    return True

class GDALDatasetContext:
    """Context manager for GDAL datasets to ensure proper cleanup."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        if self.dataset is None:
            raise ValueError("Failed to open dataset")
        return self.dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.dataset:
                self.dataset.FlushCache()
                self.dataset = None
        except:
            pass  # Ignore cleanup errors
        return False  # Re-raise any exceptions

class SamplingWindowGenerator:
    """
    Class responsible for generating clipping window coordinates using different strategies.
    Uses a base raster to define project properties (CRS, extent, pixel size).
    """
    
    def __init__(self, base_raster_path):
        """
        Initialize the clipping window generator with a base raster.
        
        Args:
            base_raster_path (str): Path to base raster file that defines project properties
        """
        self.base_raster_path = base_raster_path
        self.ds = None
        self.spatial_ref = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.extent = None
        
        try:
            self.ds = self._load_base_raster()
            self.spatial_ref = self._get_spatial_ref()
            self.pixel_size_x, self.pixel_size_y = self._get_pixel_size()
            self.extent = self._get_extent()
        except Exception as e:
            raise ValueError(f"Failed to initialize SamplingWindowGenerator: {str(e)}")
        
    def _load_base_raster(self):
        """Load and validate base raster."""
        if not os.path.exists(self.base_raster_path):
            raise ValueError(f"Base raster file does not exist: {self.base_raster_path}")
            
        ds = gdal.Open(self.base_raster_path, gdal.GA_ReadOnly)
        if not ds:
            raise ValueError(f"Could not open base raster file: {self.base_raster_path}")
        return ds
        
    def _get_spatial_ref(self):
        """Get spatial reference from base raster."""
        try:
            spatial_ref = osr.SpatialReference()
            projection = self.ds.GetProjection()
            if not projection:
                raise ValueError("Base raster has no projection information")
            spatial_ref.ImportFromWkt(projection)
            return spatial_ref
        except Exception as e:
            raise ValueError(f"Failed to get spatial reference: {str(e)}")
        
    def _get_pixel_size(self):
        """Get pixel size from base raster."""
        try:
            gt = self.ds.GetGeoTransform()
            if not gt:
                raise ValueError("Base raster has no geotransform")
            return abs(gt[1]), abs(gt[5])
        except Exception as e:
            raise ValueError(f"Failed to get pixel size: {str(e)}")
        
    def _get_extent(self):
        """Get extent from base raster."""
        try:
            gt = self.ds.GetGeoTransform()
            width = self.ds.RasterXSize
            height = self.ds.RasterYSize
            
            minx = gt[0]
            maxy = gt[3]
            maxx = gt[0] + width * gt[1]
            miny = gt[3] + height * gt[5]
            
            return (minx, miny, maxx, maxy)
        except Exception as e:
            raise ValueError(f"Failed to get extent: {str(e)}")

    def _validate_vector_crs(self, vector_path):
        """
        Validate that vector CRS matches the project CRS.
        
        Args:
            vector_path (str): Path to vector file
            
        Raises:
            ValueError: If CRS doesn't match or file can't be opened
        """
        if not os.path.exists(vector_path):
            raise ValueError(f"Vector file does not exist: {vector_path}")
            
        ds = ogr.Open(vector_path, 0)  # Read-only
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        try:
            layer = ds.GetLayer()
            if not layer:
                raise ValueError("Could not get layer from vector file")
                
            vector_sr = layer.GetSpatialRef()
            if not vector_sr:
                print(f"Warning: Vector file has no spatial reference: {vector_path}")
                return  # Skip CRS validation if no SRS
                
            if not self.spatial_ref.IsSame(vector_sr):
                raise ValueError("Vector CRS does not match project CRS")
                
        finally:
            ds = None

    def get_uuid_values(self, vector_path, uuid_col):
        """
        Get all UUID values from the specified column in the vector file.
        
        Args:
            vector_path (str): Path to vector file
            uuid_col (str): Name of the UUID column
            
        Returns:
            list: List of UUID values from the specified column
            
        Raises:
            ValueError: If UUID column doesn't exist or vector file can't be opened
        """
        ds = ogr.Open(vector_path, 0)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        try:
            layer = ds.GetLayer()
            layer_defn = layer.GetLayerDefn()
            
            # Verify UUID column exists
            field_idx = layer_defn.GetFieldIndex(uuid_col)
            if field_idx == -1:
                raise ValueError(f"UUID column '{uuid_col}' not found in vector file")
                
            uuid_values = []
            for feature in layer:
                uuid_val = feature.GetField(uuid_col)
                if _is_valid_id(uuid_val):  # Only append valid values
                    uuid_values.append(uuid_val)
            
            return uuid_values
        finally:
            ds = None

    def get_feature_window_coordinates(self, feature, window_size):
        """
        Generate window coordinates for a single feature based on its centroid.
        
        Args:
            feature (ogr.Feature): Input OGR feature
            window_size (float): Size of square sampling window in map units
            
        Returns:
            tuple: window_coordinates as (minx, maxy, maxx, miny)
        """
        try:
            half_size = window_size / 2
            
            geom = feature.GetGeometryRef()
            if not geom:
                raise ValueError("Feature has no geometry")
                
            centroid = geom.Centroid()
            x, y = centroid.GetX(), centroid.GetY()
            
            coordinates = (
                x - half_size,  # minx
                y + half_size,  # maxy
                x + half_size,  # maxx
                y - half_size   # miny
            )
            
            return coordinates
        except Exception as e:
            raise ValueError(f"Failed to get feature coordinates: {str(e)}")
    
    def get_centroid_coordinate_by_uuid(self, vector_path, window_size, uuid_col, uuid_val):
        """
        Get window coordinate for a specific feature based on its UUID value.
        Only processes features with valid (non-null) UUID values.
        
        Args:
            vector_path (str): Path to vector file
            window_size (int): Size of square sampling window in pixels
            uuid_col (str): Name of the UUID column
            uuid_val (str): UUID value to search for
            
        Returns:
            tuple: (window_coordinates, name) or (None, None) if UUID not found or invalid
                - window_coordinates: tuple of (minx, maxy, maxx, miny) coordinates
                - name: the matching UUID value as a string
        """
        # Check if the UUID value itself is valid
        if not _is_valid_id(uuid_val):
            return None, None
            
        # Validate CRS
        self._validate_vector_crs(vector_path)
        
        # Convert window size to map units
        window_size = window_size * self.pixel_size_x
        
        ds = ogr.Open(vector_path, 0)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        try:
            layer = ds.GetLayer()
            
            # Verify UUID column exists
            layer_defn = layer.GetLayerDefn()
            if layer_defn.GetFieldIndex(uuid_col) == -1:
                raise ValueError(f"UUID column '{uuid_col}' not found in vector file")
            
            # Create attribute filter using the UUID column
            layer.SetAttributeFilter(f"{uuid_col} = '{uuid_val}'")
            
            feature = layer.GetNextFeature()
            if feature is None:
                return None, None
                
            coords = self.get_feature_window_coordinates(feature, window_size)
            name = str(feature.GetField(uuid_col))
            
            return coords, name
        finally:
            ds = None

    def get_centroid_coordinates_by_uuids(self, vector_path, window_size, uuid_col, uuids_val):
        """
        Get window coordinates for multiple features based on a list of UUID values.
        Only processes features with valid (non-null) UUID values.
        """
        # Filter out invalid UUIDs first
        valid_uuids = [uuid for uuid in uuids_val if _is_valid_id(uuid)]
        invalid_count = len(uuids_val) - len(valid_uuids)
        
        if invalid_count > 0:
            print(f"Skipping {invalid_count} invalid/null UUID values")
        
        if not valid_uuids:
            print("No valid UUID values found")
            return tuple(), tuple()
        
        # Validate CRS
        self._validate_vector_crs(vector_path)
        
        # Convert window size to map units
        window_size = window_size * self.pixel_size_x
        
        ds = ogr.Open(vector_path, 0)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        try:
            layer = ds.GetLayer()
            
            # Verify UUID column exists
            layer_defn = layer.GetLayerDefn()
            if layer_defn.GetFieldIndex(uuid_col) == -1:
                raise ValueError(f"UUID column '{uuid_col}' not found in vector file")
            
            coordinates = []
            names = []
            
            # Create an SQL-like string for all valid UUIDs
            uuid_str_list = [f"'{uuid}'" for uuid in valid_uuids]
            uuid_filter = f"{uuid_col} IN ({','.join(uuid_str_list)})"
            layer.SetAttributeFilter(uuid_filter)
            
            # Dictionary to track found UUIDs
            found_uuids = set()
            
            # Get coordinates for each feature
            for feature in layer:
                uuid_val = str(feature.GetField(uuid_col))
                coords = self.get_feature_window_coordinates(feature, window_size)
                
                coordinates.append(coords)
                names.append(uuid_val)
                found_uuids.add(uuid_val)
            
            # Check for any UUIDs not found
            not_found = [uuid for uuid in valid_uuids if str(uuid) not in found_uuids]
            if not_found:
                print(f"Warning: The following valid UUID values were not found: {not_found}")
            
            if not coordinates:
                return tuple(), tuple()
            
            return tuple(coordinates), tuple(names)
        finally:
            ds = None

    def get_sliding_window_coordinates(self, window_size, stride=None):
        """
        Generate sliding window coordinates based on the base raster properties.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        
        if stride is None:
            stride = int(window_size * 0.8)
        elif not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer")
            
        # Convert to map units
        window_size = window_size * self.pixel_size_x
        stride = stride * self.pixel_size_x
        
        coordinates = []
        names = []
        x_count = 0
        
        minx, miny, maxx, maxy = self.extent
        x_steps = np.arange(minx, maxx - window_size + stride, stride)
        y_steps = np.arange(miny, maxy - window_size + stride, stride)
        
        for x in x_steps:
            y_count = 0
            for y in y_steps:
                window_minx = x
                window_miny = y
                window_maxx = x + window_size
                window_maxy = y + window_size
                
                if window_maxx <= maxx and window_maxy <= maxy:
                    coordinates.append((window_minx, window_maxy, window_maxx, window_miny))
                    names.append(f"x{x_count}_y{y_count}")
                y_count += 1
            x_count += 1
            
        return tuple(coordinates), tuple(names)

    def get_centroid_coordinates(self, vector_path, window_size, naming_field=None):
        """
        Generate window coordinates centered on vector feature centroids.
        Only processes features with valid ID values, skips null/None IDs.
        """
        # Validate CRS
        self._validate_vector_crs(vector_path)
        
        # Convert window size to map units
        window_size = window_size * self.pixel_size_x
        
        coordinates = []
        names = []
        skipped_count = 0
        total_features = 0
        
        ds = ogr.Open(vector_path, 0)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        try:
            layer = ds.GetLayer()
            total_features = layer.GetFeatureCount()
            
            # Check if naming field exists
            field_exists = False
            if naming_field:
                layer_defn = layer.GetLayerDefn()
                for i in range(layer_defn.GetFieldCount()):
                    if layer_defn.GetFieldDefn(i).GetName() == naming_field:
                        field_exists = True
                        break
                
                if not field_exists:
                    print(f"Warning: Naming field '{naming_field}' not found. Using FID for naming.")
            
            for feature in layer:
                try:
                    # Get name first to check if it's valid
                    if naming_field and field_exists:
                        field_value = feature.GetField(naming_field)
                        
                        # Check if ID is valid
                        if not _is_valid_id(field_value):
                            skipped_count += 1
                            continue  # Skip this feature entirely
                        
                        name = _sanitize_filename(field_value)
                    else:
                        # Using FID, which is always valid
                        name = f"FID_{feature.GetFID()}"
                    
                    # Get coordinates only for valid features
                    geom = feature.GetGeometryRef()
                    if not geom:
                        print(f"Warning: Feature {feature.GetFID()} has no geometry, skipping")
                        skipped_count += 1
                        continue
                        
                    centroid = geom.Centroid()
                    x, y = centroid.GetX(), centroid.GetY()
                    
                    half_size = window_size / 2
                    coords = (
                        x - half_size,  # minx
                        y + half_size,  # maxy
                        x + half_size,  # maxx
                        y - half_size   # miny
                    )
                    
                    coordinates.append(coords)
                    names.append(name)
                    
                except Exception as e:
                    print(f"Warning: Error processing feature {feature.GetFID()}: {e}")
                    skipped_count += 1
                    continue
            
            # Report results
            processed_count = len(coordinates)
            print(f"\nFeature Processing Summary:")
            print(f"  Total features in vector file: {total_features}")
            print(f"  Features with valid IDs: {processed_count}")
            print(f"  Features skipped (null/invalid IDs): {skipped_count}")
            
            if skipped_count > 0:
                print(f"  → Skipped {skipped_count} features with null/None/empty ID values")
            
            if processed_count == 0:
                raise ValueError("No features with valid IDs found for processing!")
                
            print(f"  → Proceeding with {processed_count} valid features\n")
            
            return tuple(coordinates), tuple(names)
            
        finally:
            ds = None
    
    def __del__(self):
        """Cleanup GDAL resources."""
        try:
            if hasattr(self, 'ds') and self.ds is not None:
                self.ds = None
        except:
            pass

class BaseSampler:
    """Base class for clipping raster and vector data using coordinates."""
    
    def __init__(self, base_raster_path):
        """Initialize with a base raster for CRS and extent validation."""
        self.base_raster_path = base_raster_path
        self.ds = None
        self.spatial_ref = None
        self.extent = None
        self.pixel_size = None
        
        try:
            if not os.path.exists(base_raster_path):
                raise ValueError(f"Base raster file does not exist: {base_raster_path}")
                
            self.ds = gdal.Open(base_raster_path, gdal.GA_ReadOnly)
            if not self.ds:
                raise ValueError(f"Could not open base raster: {base_raster_path}")
                
            self.spatial_ref = self._get_spatial_ref()
            self.extent = self._get_extent()
            self.pixel_size = abs(self.ds.GetGeoTransform()[1])
        except Exception as e:
            raise ValueError(f"Failed to initialize BaseSampler: {str(e)}")

    def _get_spatial_ref(self):
        """Get spatial reference from base raster."""
        try:
            spatial_ref = osr.SpatialReference()
            projection = self.ds.GetProjection()
            if not projection:
                raise ValueError("Base raster has no projection information")
            spatial_ref.ImportFromWkt(projection)
            return spatial_ref
        except Exception as e:
            raise ValueError(f"Failed to get spatial reference: {str(e)}")
        
    def _get_extent(self):
        """Get extent from base raster."""
        try:
            gt = self.ds.GetGeoTransform()
            width = self.ds.RasterXSize
            height = self.ds.RasterYSize
            
            minx = gt[0]
            maxy = gt[3]
            maxx = gt[0] + width * gt[1]
            miny = gt[3] + height * gt[5]
            
            return (minx, miny, maxx, maxy)
        except Exception as e:
            raise ValueError(f"Failed to get extent: {str(e)}")

    def _validate_coordinates(self, coords):
        """Validate if coordinates fall within project extent."""
        try:
            minx, maxy, maxx, miny = coords
            extent_minx, extent_miny, extent_maxx, extent_maxy = self.extent
            
            return (minx >= extent_minx and maxx <= extent_maxx and
                    miny >= extent_miny and maxy <= extent_maxy)
        except Exception as e:
            print(f"Warning: Could not validate coordinates: {str(e)}")
            return True  # Allow processing to continue

    def _validate_raster_crs(self, raster_path):
        """Validate raster CRS matches project CRS."""
        if not os.path.exists(raster_path):
            raise ValueError(f"Raster file does not exist: {raster_path}")
            
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if not ds:
            raise ValueError(f"Could not open raster: {raster_path}")
            
        try:
            raster_sr = osr.SpatialReference()
            projection = ds.GetProjection()
            if projection:
                raster_sr.ImportFromWkt(projection)
                
                if not self.spatial_ref.IsSame(raster_sr):
                    print(f"Warning: Raster CRS may not match project CRS")
            else:
                print(f"Warning: Raster has no projection information: {raster_path}")
        finally:
            ds = None

    def _validate_vector_crs(self, vector_path):
        """Validate vector CRS matches project CRS."""
        if not os.path.exists(vector_path):
            raise ValueError(f"Vector file does not exist: {vector_path}")
            
        ds = ogr.Open(vector_path, 0)
        if not ds:
            raise ValueError(f"Could not open vector: {vector_path}")
            
        try:
            layer = ds.GetLayer()
            vector_sr = layer.GetSpatialRef()
            
            if vector_sr and not self.spatial_ref.IsSame(vector_sr):
                print(f"Warning: Vector CRS may not match project CRS")
        finally:
            ds = None

    def clip_raster(self, raster_path, coords, output_format="array", name=None, save=None, prefix="", output_dir=None, custom_save_path=None):
        """
        Clip raster data for given coordinates with improved error handling.
        """
        try:
            # Validate inputs
            if not self._validate_coordinates(coords):
                print(f"Warning: Coordinates may be outside project extent")
            
            if save and not custom_save_path and (name is None or output_dir is None):
                raise ValueError("When save=True, either custom_save_path or both name and output_dir must be provided")

            # Handle saving with simple naming
            save_path = custom_save_path
            if save and not save_path:
                os.makedirs(output_dir, exist_ok=True)
                sanitized_name = _sanitize_filename(name)
                save_path = os.path.join(output_dir, f"{prefix}{sanitized_name}.tif")
            
            # Extract coordinates
            minx, maxy, maxx, miny = coords
            
            # Calculate dimensions
            width = int((maxx - minx) / self.pixel_size)
            height = int((maxy - miny) / self.pixel_size)
            
            # Ensure minimum dimensions
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")
            
            # Use nested context managers for proper resource cleanup
            with GDALDatasetContext(gdal.Open(raster_path, gdal.GA_ReadOnly)) as src_ds:
                if output_format == "array":
                    # Set up warp options for array output
                    warp_options = gdal.WarpOptions(
                        format='MEM',
                        outputBounds=[minx, miny, maxx, maxy],
                        width=width,
                        height=height,
                        resampleAlg=gdal.GRA_Bilinear,
                        errorThreshold=0.125,
                        multithread=False  # Disable for stability
                    )
                    
                    with GDALDatasetContext(gdal.Warp('', src_ds, options=warp_options)) as warped_ds:
                        array = warped_ds.ReadAsArray()
                        if save_path:
                            cv2.imwrite(save_path, array)
                        return array
                        
                else:  # geocoded
                    # Create the geocoded dataset with compression
                    creation_options = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
                    if save_path:
                        output_path = save_path
                    else:
                        output_path = ''  # Create in memory
                        
                    warp_options = gdal.WarpOptions(
                        format='GTiff' if save_path else 'MEM',
                        outputBounds=[minx, miny, maxx, maxy],
                        width=width,
                        height=height,
                        resampleAlg=gdal.GRA_Bilinear,
                        errorThreshold=0.125,
                        multithread=False,  # Disable for stability
                        creationOptions=creation_options if save_path else None
                    )
                    
                    # Create the warped dataset
                    warped_ds = gdal.Warp(output_path, src_ds, options=warp_options)
                    
                    if warped_ds is None:
                        raise RuntimeError("Failed to create warped dataset")
                    
                    if save_path:
                        warped_ds.FlushCache()
                        warped_ds = None  # Close the file
                        return save_path
                    
                    return warped_ds

        except Exception as e:
            raise RuntimeError(f"Error clipping raster: {str(e)}")

    def clip_vector(self, vector_path, coords, pixel_size=None, burn_value=1.0, 
                    output_format="array", name=None, save=None, prefix="", 
                    output_dir=None, filter_col=None, custom_save_path=None):
        """
        Clip vector data for given coordinates with improved error handling.
        """
        try:
            # Validate inputs
            if not self._validate_coordinates(coords):
                print(f"Warning: Coordinates may be outside project extent")

            if save and not custom_save_path and (name is None or output_dir is None):
                raise ValueError("When save=True, either custom_save_path or both name and output_dir must be provided")

            # Handle saving with simple naming
            save_path = custom_save_path
            if save and not save_path:
                os.makedirs(output_dir, exist_ok=True)
                sanitized_name = _sanitize_filename(name)
                save_path = os.path.join(output_dir, f"{prefix}{sanitized_name}.tif")

            # Set pixel size
            pixel_size = pixel_size or self.pixel_size
            minx, maxy, maxx, miny = coords
            
            width = int((maxx - minx) / pixel_size)
            height = int((maxy - miny) / pixel_size)
            
            # Ensure minimum dimensions
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")
            
            # Create spatial filter polygon
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(minx, miny)
            ring.AddPoint(maxx, miny)
            ring.AddPoint(maxx, maxy)
            ring.AddPoint(minx, maxy)
            ring.AddPoint(minx, miny)
            
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            
            # Process vector data
            with GDALDatasetContext(ogr.Open(vector_path, 0)) as vector_ds:
                layer = vector_ds.GetLayer()
                layer.SetSpatialFilter(poly)

                # Apply attribute filter if filter_col is provided
                if filter_col is not None:
                    if name is None:
                        raise ValueError("name parameter is required when using filter_col")
                    
                    layer_defn = layer.GetLayerDefn()
                    if layer_defn.GetFieldIndex(filter_col) == -1:
                        raise ValueError(f"Filter column '{filter_col}' not found in vector file")
                    
                    layer.SetAttributeFilter(f"{filter_col} = '{name}'")

                # Create memory dataset
                with GDALDatasetContext(
                    gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Float32)
                ) as target_ds:
                    # Set up target dataset
                    target_ds.GetRasterBand(1).Fill(0)
                    transform = (minx, pixel_size, 0, maxy, 0, -pixel_size)
                    target_ds.SetGeoTransform(transform)
                    target_ds.SetProjection(self.spatial_ref.ExportToWkt())

                    # Perform rasterization
                    rasterize_options = ['ALL_TOUCHED=TRUE']
                    err = gdal.RasterizeLayer(target_ds, [1], layer, 
                                            burn_values=[burn_value],
                                            options=rasterize_options)

                    if err != 0:
                        raise RuntimeError(f"Rasterization failed with error code {err}")

                    target_ds.FlushCache()
                    
                    if output_format == "array":
                        array = target_ds.ReadAsArray()
                        if array is None:
                            raise RuntimeError("Failed to read array from rasterized data")
                        if save_path:
                            cv2.imwrite(save_path, array)
                        return array
                    else:  # geocoded
                        if save_path:
                            # Create a GTiff copy with compression
                            driver = gdal.GetDriverByName('GTiff')
                            driver.CreateCopy(
                                save_path, target_ds, 
                                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
                            )
                            return save_path
                        
                        # Return a memory copy of the result
                        return gdal.GetDriverByName('MEM').CreateCopy('', target_ds)

        except Exception as e:
            raise RuntimeError(f"Error clipping vector: {str(e)}")
        finally:
            try:
                if 'ring' in locals() and ring is not None:
                    ring = None
                if 'poly' in locals() and poly is not None:
                    poly = None
            except:
                pass

    def __del__(self):
        """Cleanup GDAL resources."""
        try:
            if hasattr(self, 'ds') and self.ds is not None:
                self.ds = None
        except:
            pass

# Helper functions for multiprocessing - must be defined after BaseSampler
def _process_raster_chunk_helper(args):
    """Helper function for raster processing that can be pickled."""
    chunk_data, raster_path, output_format, output_dir, prefix, base_raster_path = args
    try:
        coords, name = chunk_data
        
        # Use simple naming: prefix + name + .tif
        if output_format == "geocoded":
            save_path = _create_simple_filepath(output_dir, prefix, name)
        else:
            save_path = None
        
        sampler = BaseSampler(base_raster_path)
        result = sampler.clip_raster(
            raster_path, coords, output_format=output_format,
            name=name, save=True if save_path else False, 
            prefix=prefix, output_dir=output_dir,
            custom_save_path=save_path
        )
        del sampler
        return None
    except Exception as e:
        return f"Error processing raster chunk {name}: {str(e)}"

def _process_vector_chunk_helper(args):
    """Helper function for vector processing that can be pickled."""
    chunk_data, vector_path, pixel_size, burn_value, output_format, output_dir, prefix, base_raster_path, filter_col = args
    try:
        coords, name = chunk_data
        
        # Use simple naming: prefix + name + .tif
        save_path = _create_simple_filepath(output_dir, prefix, name)
        
        sampler = BaseSampler(base_raster_path)
        result = sampler.clip_vector(
            vector_path, coords, pixel_size=pixel_size, burn_value=burn_value,
            output_format=output_format, name=name, save=True,
            prefix=prefix, output_dir=output_dir, filter_col=filter_col,
            custom_save_path=save_path
        )
        del sampler
        return None
    except Exception as e:
        return f"Error processing vector chunk {name}: {str(e)}"

class Sampler(BaseSampler):
    """Class for handling both single and batch clipping operations with progress tracking."""
    
    def __init__(self, base_raster_path):
        """Initialize Sampler with base raster."""
        super().__init__(base_raster_path)

    def _validate_inputs(self, coordinates, names):
        """Validate coordinates and names inputs."""
        if not coordinates or not names:
            raise ValueError("Coordinates and names cannot be empty")
        if len(coordinates) != len(names):
            raise ValueError("Number of coordinates must match number of names")

    def _validate_workers(self, n_workers):
        """Validate and adjust number of workers."""
        max_workers = max(1, int(psutil.cpu_count() * 0.6))
        if n_workers > max_workers:
            print(f"Reducing workers from {n_workers} to {max_workers} (60% of CPU cores)")
            return max_workers
        return max(1, n_workers)
        
    def _parallel_process_raster(self, coordinates, names, raster_path, output_format, 
                               output_dir, prefix, n_workers):
        """Handle parallel processing for raster clipping with progress bar."""
        # Validate multiprocessing context before starting
        if not _validate_multiprocessing_context():
            print("Warning: Multiprocessing context validation failed. Using serial processing instead.")
            self._serial_process_raster(coordinates, names, raster_path, output_format, output_dir, prefix)
            return
            
        try:
            with mp.Pool(n_workers, initializer=_init_gdal) as pool:
                args_list = [
                    (chunk, raster_path, output_format, output_dir, prefix, self.base_raster_path)
                    for chunk in zip(coordinates, names)
                ]
                
                results = list(tqdm(
                    pool.imap_unordered(_process_raster_chunk_helper, args_list),
                    total=len(coordinates),
                    desc=f"Clipping raster with {n_workers} workers",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                ))
                
                # Filter out None results and collect errors
                errors = [r for r in results if r is not None]
                if errors:
                    print(f"Found {len(errors)} errors during processing:")
                    for i, error in enumerate(errors[:5]):  # Show first 5 errors
                        print(f"  {i+1}. {error}")
                    if len(errors) > 5:
                        print(f"  ... and {len(errors) - 5} more errors")
                    
                    # Only raise if more than 50% failed
                    if len(errors) > len(coordinates) * 0.5:
                        raise RuntimeError(f"Too many chunks failed: {len(errors)}/{len(coordinates)}")
                    else:
                        print(f"Warning: {len(errors)} chunks failed, but continuing with successful ones")
                        
        except Exception as e:
            print(f"Parallel processing failed: {str(e)}")
            print("Falling back to serial processing...")
            self._serial_process_raster(coordinates, names, raster_path, output_format, output_dir, prefix)

    def _serial_process_raster(self, coordinates, names, raster_path, output_format, 
                             output_dir, prefix):
        """Handle serial processing for raster clipping with progress bar."""
        errors = []
        for coords, name in tqdm(zip(coordinates, names), 
                               total=len(coordinates),
                               desc="Clipping raster (serial)",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            args = ((coords, name), raster_path, output_format, output_dir, prefix, self.base_raster_path)
            error = _process_raster_chunk_helper(args)
            if error:
                errors.append(error)
                
        if errors:
            print(f"Found {len(errors)} errors during serial processing:")
            for i, error in enumerate(errors[:3]):  # Show first 3 errors
                print(f"  {i+1}. {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")

    def _parallel_process_vector(self, coordinates, names, vector_path, pixel_size, 
                               burn_value, output_format, output_dir, prefix, 
                               n_workers, filter_col):
        """Handle parallel processing for vector clipping with progress bar."""
        # Validate multiprocessing context before starting
        if not _validate_multiprocessing_context():
            print("Warning: Multiprocessing context validation failed. Using serial processing instead.")
            self._serial_process_vector(coordinates, names, vector_path, pixel_size, 
                                      burn_value, output_format, output_dir, prefix, filter_col)
            return
            
        try:
            with mp.Pool(n_workers, initializer=_init_gdal) as pool:
                args_list = [
                    (chunk, vector_path, pixel_size, burn_value, output_format, output_dir, 
                     prefix, self.base_raster_path, filter_col)
                    for chunk in zip(coordinates, names)
                ]
                
                results = list(tqdm(
                    pool.imap_unordered(_process_vector_chunk_helper, args_list),
                    total=len(coordinates),
                    desc=f"Clipping vector with {n_workers} workers",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                ))
                
                errors = [r for r in results if r is not None]
                if errors:
                    print(f"Found {len(errors)} errors during processing:")
                    for i, error in enumerate(errors[:5]):  # Show first 5 errors
                        print(f"  {i+1}. {error}")
                    if len(errors) > 5:
                        print(f"  ... and {len(errors) - 5} more errors")
                    
                    # Only raise if more than 50% failed
                    if len(errors) > len(coordinates) * 0.5:
                        raise RuntimeError(f"Too many chunks failed: {len(errors)}/{len(coordinates)}")
                    else:
                        print(f"Warning: {len(errors)} chunks failed, but continuing with successful ones")
                        
        except Exception as e:
            print(f"Parallel processing failed: {str(e)}")
            print("Falling back to serial processing...")
            self._serial_process_vector(coordinates, names, vector_path, pixel_size, 
                                      burn_value, output_format, output_dir, prefix, filter_col)

    def _serial_process_vector(self, coordinates, names, vector_path, pixel_size, 
                             burn_value, output_format, output_dir, prefix, filter_col):
        """Handle serial processing for vector clipping with progress bar."""
        errors = []
        for coords, name in tqdm(zip(coordinates, names), 
                               total=len(coordinates),
                               desc="Clipping vector (serial)",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            args = ((coords, name), vector_path, pixel_size, burn_value, output_format, 
                   output_dir, prefix, self.base_raster_path, filter_col)
            error = _process_vector_chunk_helper(args)
            if error:
                errors.append(error)
                
        if errors:
            print(f"Found {len(errors)} errors during serial processing:")
            for i, error in enumerate(errors[:3]):  # Show first 3 errors
                print(f"  {i+1}. {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")

    def clip_rasters(self, raster_path, coordinates, names, output_dir, 
                    output_format="geocoded", prefix="", n_workers=1):
        """Clip raster for multiple coordinates with optional parallel processing."""
        self._validate_inputs(coordinates, names)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing {len(coordinates)} raster clips...")

        try:
            if n_workers > 1:
                n_workers = self._validate_workers(n_workers)
                self._parallel_process_raster(
                    coordinates, names, raster_path, output_format, 
                    output_dir, prefix, n_workers
                )
            else:
                self._serial_process_raster(
                    coordinates, names, raster_path, output_format, 
                    output_dir, prefix
                )
            print("\nRaster clipping completed!")
        except Exception as e:
            print(f"\nError during raster clipping: {str(e)}")
            raise

    def clip_vectors(self, vector_path, coordinates, names, output_dir,
                    pixel_size=None, burn_value=1.0, output_format="array", 
                    prefix="", n_workers=1, filter_col=None):
        """
        Clip vector for multiple coordinates with optional parallel processing and feature filtering.
        Only processes features with valid (non-null) ID values.
        
        Args:
            vector_path (str): Path to vector file to be clipped
            coordinates (tuple): Tuple of coordinate tuples (minx, maxy, maxx, miny)
            names (tuple): Tuple of names corresponding to each coordinate tuple
            output_dir (str): Directory where clipped outputs will be saved
            pixel_size (float, optional): Pixel size for rasterization. If None, uses base raster's pixel size
            burn_value (float, optional): Value to burn into output raster. Defaults to 1.0
            output_format (str, optional): Format of output, either "array" or "geocoded"
            prefix (str, optional): Prefix to add to output filenames. Defaults to ""
            n_workers (int, optional): Number of parallel workers. If > 1, enables parallel processing
            filter_col (str, optional): Column name to filter features. If provided, only features where 
                                    filter_col value matches the name parameter will be rasterized
        """
        self._validate_inputs(coordinates, names)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing {len(coordinates)} vector clips...")
        
        try:
            if n_workers > 1:
                n_workers = self._validate_workers(n_workers)
                self._parallel_process_vector(
                    coordinates, names, vector_path, pixel_size, 
                    burn_value, output_format, output_dir, prefix, 
                    n_workers, filter_col
                )
            else:
                self._serial_process_vector(
                    coordinates, names, vector_path, pixel_size, 
                    burn_value, output_format, output_dir, prefix, filter_col
                )
                
            print("\nVector clipping completed!")
            
        except Exception as e:
            print(f"\nError during vector clipping: {str(e)}")
            raise

    def __del__(self):
        """Cleanup and restore GDAL error handling."""
        try:
            super().__del__()
        except:
            pass