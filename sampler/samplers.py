"""
Samplers Module

Overview:
  This module provides robust geospatial sampling and clipping operations for raster and vector datasets,
  with comprehensive coordinate generation, data validation, and efficient memory management.

Classes:
  SamplingWindowGenerator:
      Purpose: Generates sampling window coordinates based on various strategies
      Features:
      - Center-based sampling from vector features with UUID and attribute filtering
      - Sliding window sampling with configurable window/stride sizes
      - Uses base raster for CRS, extent, and pixel size reference
      - Comprehensive coordinate validation and error handling
      - Multiple coordinate generation strategies (by UUID, centroid, sliding window)
      - Cross-dataset CRS validation

  BaseSampler:
      Purpose: Provides core clipping operations with robust resource management
      Features:
      - Raster and vector data clipping with GDAL
      - Both array (NumPy) and geocoded (GeoTIFF) output formats
      - Context-managed GDAL resource handling
      - Automatic spatial reference validation and transformation
      - GDAL-optimized operations with compression support
      - Vector rasterization with customizable burn values
      - Detailed error handling and validation

  Sampler (inherits BaseSampler):
      Purpose: Extends BaseSampler with batch processing capabilities
      Features:
      - Serial and parallel processing modes
      - Progress tracking with tqdm
      - Automatic CPU core optimization
      - Resource-efficient chunk processing
      - Feature filtering support for vector operations
      - Comprehensive error reporting

Module-Level Components:
  GDALDatasetContext:
      Purpose: Context manager for safe GDAL resource handling
      Features:
      - Automatic cleanup of GDAL datasets
      - Exception-safe resource management
      - Memory leak prevention

  Helper Functions:
      _init_gdal():
          - Configures GDAL error handling
          - Sets up multiprocessing environment
          - Manages warning suppression
      
      _process_raster_chunk_helper():
          - Handles individual raster clip operations
          - Ensures isolated resource management
      
      _process_vector_chunk_helper():
          - Manages vector clip operations
          - Maintains resource isolation

Technical Details:
  Dependencies:
      - GDAL/OGR (>= 2.0): Geospatial data operations
      - NumPy: Array processing
      - OpenCV: Image I/O
      - psutil: System resource management
      - tqdm: Progress visualization
      - multiprocessing: Parallel processing support

  Implementation Features:
      - Thread-safe operations with proper resource isolation
      - Memory-efficient processing with automatic cleanup
      - Comprehensive error handling and validation
      - Support for compressed GeoTIFF output
      - Configurable feature filtering for vector operations
      - Flexible coordinate generation strategies

Note:
  All spatial operations maintain proper CRS validation and transformation.
  Resource cleanup is handled automatically through context managers.
  Error handling includes detailed feedback for debugging.
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

# Helper functions for multiprocessing
def _init_gdal():
    """Initialize GDAL configuration for worker processes."""
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetConfigOption('CPL_DEBUG', 'OFF')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

class GDALDatasetContext:
    """Context manager for GDAL datasets to ensure proper cleanup."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        if self.dataset is None:
            raise ValueError("Failed to open dataset")
        return self.dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dataset:
            self.dataset = None
        return False  # Re-raise any exceptions

def _process_raster_chunk_helper(args):
    """Helper function for raster processing that can be pickled."""
    chunk_data, raster_path, output_format, output_dir, prefix, base_raster_path = args
    try:
        coords, name = chunk_data
        sampler = BaseSampler(base_raster_path)
        sampler.clip_raster(
            raster_path, coords, output_format=output_format,
            name=name, save=True, prefix=prefix, output_dir=output_dir
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
        sampler = BaseSampler(base_raster_path)
        sampler.clip_vector(
            vector_path, coords, pixel_size=pixel_size, burn_value=burn_value,
            output_format=output_format, name=name, save=True,
            prefix=prefix, output_dir=output_dir, filter_col=filter_col
        )
        del sampler
        return None
    except Exception as e:
        return f"Error processing vector chunk {name}: {str(e)}"
    
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
        self.ds = self._load_base_raster()
        self.spatial_ref = self._get_spatial_ref()
        self.pixel_size_x, self.pixel_size_y = self._get_pixel_size()
        self.extent = self._get_extent()
        
    def _load_base_raster(self):
        """Load and validate base raster."""
        ds = gdal.Open(self.base_raster_path)
        if not ds:
            raise ValueError(f"Could not open base raster file: {self.base_raster_path}")
        return ds
        
    def _get_spatial_ref(self):
        """Get spatial reference from base raster."""
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(self.ds.GetProjection())
        return spatial_ref
        
    def _get_pixel_size(self):
        """Get pixel size from base raster."""
        gt = self.ds.GetGeoTransform()
        return abs(gt[1]), abs(gt[5])
        
    def _get_extent(self):
        """Get extent from base raster."""
        gt = self.ds.GetGeoTransform()
        width = self.ds.RasterXSize
        height = self.ds.RasterYSize
        
        minx = gt[0]
        maxy = gt[3]
        maxx = gt[0] + width * gt[1]
        miny = gt[3] + height * gt[5]
        
        return (minx, miny, maxx, maxy)

    def _validate_vector_crs(self, vector_path):
        """
        Validate that vector CRS matches the project CRS.
        
        Args:
            vector_path (str): Path to vector file
            
        Raises:
            ValueError: If CRS doesn't match or file can't be opened
        """
        ds = ogr.Open(vector_path)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        layer = ds.GetLayer()
        vector_sr = layer.GetSpatialRef()
        
        if not self.spatial_ref.IsSame(vector_sr):
            ds = None
            raise ValueError(
                f"Vector CRS does not match project CRS"
            )
            
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
        ds = ogr.Open(vector_path)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()
        
        # Verify UUID column exists
        field_idx = layer_defn.GetFieldIndex(uuid_col)
        if field_idx == -1:
            ds = None
            raise ValueError(f"UUID column '{uuid_col}' not found in vector file")
            
        uuid_values = []
        for feature in layer:
            uuid_val = feature.GetField(uuid_col)
            if uuid_val is not None:  # Only append non-null values
                uuid_values.append(uuid_val)
        
        ds = None
        return uuid_values

    def get_feature_window_coordinates(self, feature, window_size):
        """
        Generate window coordinates for a single feature based on its centroid.
        
        Args:
            feature (ogr.Feature): Input OGR feature
            window_size (float): Size of square sampling window in map units
            
        Returns:
            tuple: window_coordinates as (minx, maxy, maxx, miny)
        """
        half_size = window_size / 2
        
        geom = feature.GetGeometryRef()
        centroid = geom.Centroid()
        x, y = centroid.GetX(), centroid.GetY()
        
        coordinates = (
            x - half_size,  # minx
            y + half_size,  # maxy
            x + half_size,  # maxx
            y - half_size   # miny
        )
        
        return coordinates
    
    def get_centroid_coordinate_by_uuid(self, vector_path, window_size, uuid_col, uuid_val):
        """
        Get window coordinate for a specific feature based on its UUID value.
        
        Args:
            vector_path (str): Path to vector file
            window_size (int): Size of square sampling window in pixels
            uuid_col (str): Name of the UUID column
            uuid_val (str): UUID value to search for
            
        Returns:
            tuple: (window_coordinates, name) or (None, None) if UUID not found
                - window_coordinates: tuple of (minx, maxy, maxx, miny) coordinates
                - name: the matching UUID value as a string
        """
        # Validate CRS
        self._validate_vector_crs(vector_path)
        
        # Convert window size to map units
        window_size = window_size * self.pixel_size_x
        
        ds = ogr.Open(vector_path)
        layer = ds.GetLayer()
        
        # Verify UUID column exists
        layer_defn = layer.GetLayerDefn()
        if layer_defn.GetFieldIndex(uuid_col) == -1:
            ds = None
            raise ValueError(f"UUID column '{uuid_col}' not found in vector file")
        
        # Create attribute filter using the UUID column
        layer.SetAttributeFilter(f"{uuid_col} = '{uuid_val}'")
        
        feature = layer.GetNextFeature()
        if feature is None:
            ds = None
            return None, None
            
        coords = self.get_feature_window_coordinates(feature, window_size)
        name = str(feature.GetField(uuid_col))  # Using 'name' instead of 'uuid_value' for consistency
        
        ds = None
        return coords, name

    def get_centroid_coordinates_by_uuids(self, vector_path, window_size, uuid_col, uuids_val):
        """
        Get window coordinates for multiple features based on a list of UUID values.
        
        Args:
            vector_path (str): Path to vector file
            window_size (int): Size of square sampling window in pixels
            uuid_col (str): Name of the UUID column
            uuids_val (list): List of UUID values to search for
            
        Returns:
            tuple: (window_coordinates, names)
                - window_coordinates: tuple of (minx, maxy, maxx, miny) coordinates
                - names: tuple of corresponding UUID values as strings
                
        Raises:
            ValueError: If UUID column doesn't exist or vector file can't be opened
            Warning: Prints warning for any UUID values not found in the vector file
        """
        # Validate CRS
        self._validate_vector_crs(vector_path)
        
        # Convert window size to map units
        window_size = window_size * self.pixel_size_x
        
        ds = ogr.Open(vector_path)
        if not ds:
            raise ValueError(f"Could not open vector file: {vector_path}")
            
        layer = ds.GetLayer()
        
        # Verify UUID column exists
        layer_defn = layer.GetLayerDefn()
        if layer_defn.GetFieldIndex(uuid_col) == -1:
            ds = None
            raise ValueError(f"UUID column '{uuid_col}' not found in vector file")
        
        coordinates = []
        names = []
        not_found = []
        
        # Create an SQL-like string for all UUIDs
        uuid_str_list = [f"'{uuid}'" for uuid in uuids_val]
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
        not_found = [uuid for uuid in uuids_val if str(uuid) not in found_uuids]
        if not_found:
            print(f"Warning: The following UUID values were not found: {not_found}")
        
        ds = None
        
        if not coordinates:
            return tuple(), tuple()
        
        return tuple(coordinates), tuple(names)

    def get_sliding_window_coordinates(self, window_size, stride=None):
        """
        Generate sliding window coordinates based on the base raster properties.
        
        Args:
            window_size (int): Size of square sampling window in pixels
            stride (int, optional): Stride size in pixels. 
                                    If None, defaults to 80% of window size.
            
        Returns:
            tuple: (window_coordinates, window_names)
                - window_coordinates: tuple of (minx, maxy, maxx, miny) coordinates
                - window_names: tuple of string names in format "x{x_count}_y{y_count}"
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
        
        Args:
            vector_path (str): Path to vector file
            window_size (int): Size of square sampling window in pixels
            naming_field (str, optional): Field to use for naming. If None, uses
                                          FID (Feature ID) for naming
            
        Returns:
            tuple: (window_coordinates, window_names)
                - window_coordinates: tuple of (minx, maxy, maxx, miny) coordinates
                - window_names: tuple of string names from features
        """
        # Validate CRS
        self._validate_vector_crs(vector_path)
        
        # Convert window size to map units
        window_size = window_size * self.pixel_size_x
        
        coordinates = []
        names = []
        
        ds = ogr.Open(vector_path)
        layer = ds.GetLayer()
        
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
            # Get coordinates using the new method
            coords = self.get_feature_window_coordinates(feature, window_size)
            
            # Get name from field or use FID
            if naming_field and field_exists:
                name = str(feature.GetField(naming_field))
            else:
                name = f"FID_{feature.GetFID()}"
            
            coordinates.append(coords)
            names.append(name)
            
        ds = None
    
        return tuple(coordinates), tuple(names)
    
    def __del__(self):
        """Cleanup GDAL resources."""
        if hasattr(self, 'ds') and self.ds is not None:
            self.ds = None

class BaseSampler:
    """Base class for clipping raster and vector data using coordinates."""
    
    def __init__(self, base_raster_path):
        """Initialize with a base raster for CRS and extent validation."""
        self.base_raster_path = base_raster_path
        self.ds = gdal.Open(base_raster_path)
        if not self.ds:
            raise ValueError(f"Could not open base raster: {base_raster_path}")
            
        self.spatial_ref = self._get_spatial_ref()
        self.extent = self._get_extent()
        self.pixel_size = abs(self.ds.GetGeoTransform()[1])

    def _get_spatial_ref(self):
        """Get spatial reference from base raster."""
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(self.ds.GetProjection())
        return spatial_ref
        
    def _get_extent(self):
        """Get extent from base raster."""
        gt = self.ds.GetGeoTransform()
        width = self.ds.RasterXSize
        height = self.ds.RasterYSize
        
        minx = gt[0]
        maxy = gt[3]
        maxx = gt[0] + width * gt[1]
        miny = gt[3] + height * gt[5]
        
        return (minx, miny, maxx, maxy)

    def _validate_coordinates(self, coords):
        """Validate if coordinates fall within project extent."""
        minx, maxy, maxx, miny = coords
        extent_minx, extent_miny, extent_maxx, extent_maxy = self.extent
        
        return (minx >= extent_minx and maxx <= extent_maxx and
                miny >= extent_miny and maxy <= extent_maxy)

    def _validate_raster_crs(self, raster_path):
        """Validate raster CRS matches project CRS."""
        ds = gdal.Open(raster_path)
        if not ds:
            raise ValueError(f"Could not open raster: {raster_path}")
            
        raster_sr = osr.SpatialReference()
        raster_sr.ImportFromWkt(ds.GetProjection())
        
        if not self.spatial_ref.IsSame(raster_sr):
            ds = None
            raise ValueError("Raster CRS does not match project CRS")
        ds = None

    def _validate_vector_crs(self, vector_path):
        """Validate vector CRS matches project CRS."""
        ds = ogr.Open(vector_path)
        if not ds:
            raise ValueError(f"Could not open vector: {vector_path}")
            
        layer = ds.GetLayer()
        vector_sr = layer.GetSpatialRef()
        
        if not self.spatial_ref.IsSame(vector_sr):
            ds = None
            raise ValueError("Vector CRS does not match project CRS")
        ds = None

    def _save_array(self, array, save_path):
        """Save numpy array as GeoTiff using OpenCV."""
        if not save_path:
            raise ValueError("save_path is required")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, array)

    def _save_geocoded(self, ds, save_path):
        """Save gdal dataset as geocoded GeoTiff."""
        if not save_path:
            raise ValueError("save_path is required")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gdal.GetDriverByName('GTiff').CreateCopy(save_path, ds)

    def _clip_vector_to_coords(self, vector_path, coords):
        """Clip vector to coordinates."""
        minx, maxy, maxx, miny = coords
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(minx, miny)
        ring.AddPoint(maxx, miny)
        ring.AddPoint(maxx, maxy)
        ring.AddPoint(minx, maxy)
        ring.AddPoint(minx, miny)
        
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        vector_ds = ogr.Open(vector_path)
        layer = vector_ds.GetLayer()
        layer.SetSpatialFilter(poly)
        return vector_ds

    def _rasterize_vector_clip(self, vector_ds, coords, pixel_size, burn_value=1.0):
        """Rasterize clipped vector data."""
        minx, maxy, maxx, miny = coords
        
        width = int((maxx - minx) / pixel_size)
        height = int((maxy - miny) / pixel_size)
        
        target_ds = gdal.GetDriverByName('MEM').Create(
            '', width, height, 1, gdal.GDT_Float32)
        
        target_ds.SetGeoTransform((minx, pixel_size, 0, maxy, 0, -pixel_size))
        target_ds.SetProjection(self.spatial_ref.ExportToWkt())
        
        gdal.RasterizeLayer(target_ds, [1], vector_ds.GetLayer(),
                           burn_values=[burn_value])
        
        return target_ds

    def clip_raster(self, raster_path, coords, output_format="array", name=None, save=None, prefix="", output_dir=None):
        """
        Clip raster data for given coordinates with improved GDAL options handling.
        
        Args:
            raster_path (str): Path to raster file
            coords (tuple): (minx, magy, maxx, miny) coordinates
            output_format (str): 'array' or 'geocoded'
            name (str, optional): Name for output file if saving
            save (bool, optional): Whether to save the output. If True, requires name and output_dir
            prefix (str, optional): Prefix to add to output filename. Defaults to ""
            output_dir (str, optional): Directory to save output if save is True
            
        Returns:
            numpy.ndarray if output_format is 'array'
            gdal.Dataset if output_format is 'geocoded'
        """
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        try:
            # Validate inputs
            self._validate_raster_crs(raster_path)
            if not self._validate_coordinates(coords):
                raise ValueError("Coordinates outside project extent")
            
            if save and (name is None or output_dir is None):
                raise ValueError("When save=True, both name and output_dir must be provided")

            # Handle saving
            save_path = None
            if save:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{prefix}{name}.tif")
            
            # Extract coordinates
            minx, maxy, maxx, miny = coords
            
            # Calculate dimensions
            width = int((maxx - minx) / self.pixel_size)
            height = int((maxy - miny) / self.pixel_size)
            
            # Use nested context managers for proper resource cleanup
            with GDALDatasetContext(gdal.Open(raster_path)) as src_ds:
                if output_format == "array":
                    # Set up warp options for array output
                    warp_options = gdal.WarpOptions(
                        format='MEM',
                        outputBounds=[minx, miny, maxx, maxy],
                        width=width,
                        height=height,
                        resampleAlg=gdal.GRA_Bilinear,
                        srcSRS=self.spatial_ref.ExportToWkt(),
                        dstSRS=self.spatial_ref.ExportToWkt(),
                        multithread=True
                    )
                    
                    with GDALDatasetContext(gdal.Warp('', src_ds, options=warp_options)) as warped_ds:
                        array = warped_ds.ReadAsArray()
                        if save_path:
                            cv2.imwrite(save_path, array)
                        return array
                        
                else:  # geocoded
                    # Create the geocoded dataset with compression
                    creation_options = ['COMPRESS=LZW']
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
                        srcSRS=self.spatial_ref.ExportToWkt(),
                        dstSRS=self.spatial_ref.ExportToWkt(),
                        multithread=True,
                        creationOptions=creation_options if save_path else None
                    )
                    
                    # Create the warped dataset
                    warped_ds = gdal.Warp(output_path, src_ds, options=warp_options)
                    
                    if warped_ds is None:
                        raise RuntimeError("Failed to create warped dataset")
                    
                    # If we created it in memory and need to save, save a copy
                    if save_path and not output_path:
                        gdal.GetDriverByName('GTiff').CreateCopy(
                            save_path, 
                            warped_ds, 
                            options=creation_options
                        )
                    
                    return warped_ds

        finally:
            gdal.PopErrorHandler()

    def clip_vector(self, vector_path, coords, pixel_size=None, burn_value=1.0, 
                    output_format="array", name=None, save=None, prefix="", 
                    output_dir=None, filter_col=None):
        """
        Clip vector data for given coordinates with optional feature filtering.
        
        Args:
            vector_path (str): Path to vector file
            coords (tuple): (minx, magy, maxx, miny) coordinates
            pixel_size (float, optional): Pixel size for rasterization
            burn_value (float): Value to burn into output raster
            output_format (str): 'array' or 'geocoded'
            name (str, optional): Name for output file if saving
            save (bool, optional): Whether to save the output. If True, requires name and output_dir
            prefix (str, optional): Prefix to add to output filename. Defaults to ""
            output_dir (str, optional): Directory to save output if save is True
            filter_col (str, optional): Column name to filter features. If provided,
                                      only features where filter_col value matches
                                      the name parameter will be rasterized
        """
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        try:
            # Validate inputs
            self._validate_vector_crs(vector_path)
            if not self._validate_coordinates(coords):
                raise ValueError("Coordinates outside project extent")

            if save and (name is None or output_dir is None):
                raise ValueError("When save=True, both name and output_dir must be provided")

            # Handle saving
            save_path = None
            if save:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{prefix}{name}.tif")

            # Set pixel size
            pixel_size = pixel_size or self.pixel_size
            minx, maxy, maxx, miny = coords
            
            width = int((maxx - minx) / pixel_size)
            height = int((maxy - miny) / pixel_size)
            
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
            with GDALDatasetContext(ogr.Open(vector_path)) as vector_ds:
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
                        if save_path:
                            cv2.imwrite(save_path, array)
                        return array
                    else:  # geocoded
                        if not save_path:
                            raise ValueError("save_path required for geocoded output")
                        
                        # Create a GTiff copy with compression
                        gdal.GetDriverByName('GTiff').CreateCopy(
                            save_path, target_ds, 
                            options=['COMPRESS=LZW']
                        )
                        
                        # Return a memory copy of the result
                        return gdal.GetDriverByName('MEM').CreateCopy('', target_ds)

        finally:
            gdal.PopErrorHandler()
            if ring is not None:
                ring = None
            if poly is not None:
                poly = None

    def __del__(self):
        """Cleanup GDAL resources."""
        if hasattr(self, 'ds'):
            self.ds = None

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
        max_workers = int(psutil.cpu_count() * 0.6)
        if n_workers > max_workers:
            return max_workers
        return max_workers if n_workers < 1 else n_workers
        
    def _parallel_process_raster(self, coordinates, names, raster_path, output_format, 
                               output_dir, prefix, n_workers):
        """Handle parallel processing for raster clipping with progress bar."""
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
                for error in errors:
                    print(error)
                raise RuntimeError("Some chunks failed to process")

    def _serial_process_raster(self, coordinates, names, raster_path, output_format, 
                             output_dir, prefix):
        """Handle serial processing for raster clipping with progress bar."""
        for coords, name in tqdm(zip(coordinates, names), 
                               total=len(coordinates),
                               desc="Clipping raster",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            args = ((coords, name), raster_path, output_format, output_dir, prefix, self.base_raster_path)
            error = _process_raster_chunk_helper(args)
            if error:
                print(error)
                raise RuntimeError("Failed to process chunk")

    def _parallel_process_vector(self, coordinates, names, vector_path, pixel_size, 
                               burn_value, output_format, output_dir, prefix, 
                               n_workers, filter_col):
        """Handle parallel processing for vector clipping with progress bar."""
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
                for error in errors:
                    print(error)
                raise RuntimeError("Some chunks failed to process")

    def _serial_process_vector(self, coordinates, names, vector_path, pixel_size, 
                             burn_value, output_format, output_dir, prefix, filter_col):
        """Handle serial processing for vector clipping with progress bar."""
        for coords, name in tqdm(zip(coordinates, names), 
                               total=len(coordinates),
                               desc="Clipping vector",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            args = ((coords, name), vector_path, pixel_size, burn_value, output_format, 
                   output_dir, prefix, self.base_raster_path, filter_col)
            error = _process_vector_chunk_helper(args)
            if error:
                print(error)
                raise RuntimeError("Failed to process chunk")

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
        
        Args:
            vector_path (str): Path to vector file to be clipped
            coordinates (tuple): Tuple of coordinate tuples (minx, magy, maxx, miny)
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
                
                with mp.Pool(n_workers, initializer=_init_gdal) as pool:
                    args_list = [
                        ((coords, name), vector_path, pixel_size, burn_value,
                         output_format, output_dir, prefix, self.base_raster_path, filter_col)
                        for coords, name in zip(coordinates, names)
                    ]
                    
                    results = list(tqdm(
                        pool.imap_unordered(_process_vector_chunk_helper, args_list),
                        total=len(coordinates),
                        desc=f"Clipping vector with {n_workers} workers",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    ))
                    
                    errors = [r for r in results if r is not None]
                    if errors:
                        for error in errors:
                            print(error)
                        raise RuntimeError("Some chunks failed to process")
            else:
                for coords, name in tqdm(zip(coordinates, names), 
                                    total=len(coordinates),
                                    desc="Clipping vector",
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                    args = ((coords, name), vector_path, pixel_size, burn_value, 
                           output_format, output_dir, prefix, self.base_raster_path, filter_col)
                    error = _process_vector_chunk_helper(args)
                    if error:
                        print(error)
                        raise RuntimeError("Failed to process chunk")
                
            print("\nVector clipping completed!")
            
        except Exception as e:
            print(f"\nError during vector clipping: {str(e)}")
            raise

    def __del__(self):
        """Cleanup and restore GDAL error handling."""
        super().__del__()
        gdal.PopErrorHandler()