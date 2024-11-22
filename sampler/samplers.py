"""
Samplers Module

Overview:
   This module implements geospatial sampling operations for raster and vector datasets,
   providing both coordinate generation and data clipping functionality with parallel 
   processing support.

Classes:
   SamplingWindowGenerator:
       Purpose: Generates sampling window coordinates for clipping operations
       Features:
       - Sliding window sampling with configurable window/stride sizes
       - Centroid-based sampling from vector features
       - Uses base raster for CRS, extent, and pixel size reference

   BaseSampler:
       Purpose: Provides core clipping operations
       Features:
       - Raster and vector data clipping support
       - Geocoded (GeoTIFF) and array output formats
       - Spatial reference validation
       - GDAL-optimized operations with compression
       - Automated resource cleanup

   Sampler (inherits BaseSampler):
       Purpose: Adds batch processing capabilities
       Features:
       - Parallel processing with worker management
       - Progress tracking via tqdm
       - Comprehensive error handling
       - Automatic CPU core optimization
       - Memory-efficient chunk processing

Module-Level Helper Functions:
   _init_gdal():
       Purpose: GDAL configuration for multiprocessing
       - Configures error handling and warning suppression
       - Ensures proper GDAL setup in worker processes
       Location: Module-level for multiprocessing compatibility

   _process_raster_chunk_helper(args):
       Purpose: Single raster clip operation processing
       - Maintains picklability with argument tuples
       - Isolates GDAL resources per chunk
       - Handles cleanup and error reporting
       Location: Module-level for multiprocessing pool operations

   _process_vector_chunk_helper(args):
       Purpose: Single vector clip operation processing
       - Matches raster helper structure
       - Handles vector-specific parameters
       - Manages resource isolation
       Location: Module-level for multiprocessing support

Technical Dependencies:
   - GDAL/OGR: Geospatial operations
   - NumPy: Numerical computations
   - OpenCV: Array handling
   - psutil: CPU management
   - tqdm: Progress visualization
   - multiprocessing: Parallel execution

Implementation Notes:
   - Helper functions enable thread-safe parallel processing
   - Each process maintains isolated GDAL context
   - Comprehensive error handling and reporting
   - Efficient resource management
   - Supports both serial and parallel modes
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

def _process_raster_chunk_helper(args):
    """Helper function for raster processing that can be pickled."""
    chunk_data, raster_path, output_format, output_dir, prefix, base_raster_path = args
    try:
        coords, name = chunk_data
        sampler = BaseSampler(base_raster_path)
        save_path = os.path.join(output_dir, f"{prefix}{name}.tif")
        sampler.clip_raster(raster_path, coords, output_format, save_path)
        del sampler
        return None
    except Exception as e:
        return f"Error processing raster chunk {name}: {str(e)}"

def _process_vector_chunk_helper(args):
    """Helper function for vector processing that can be pickled."""
    chunk_data, vector_path, pixel_size, burn_value, output_format, output_dir, prefix, base_raster_path = args
    try:
        coords, name = chunk_data
        sampler = BaseSampler(base_raster_path)
        save_path = os.path.join(output_dir, f"{prefix}{name}.tif")
        sampler.clip_vector(vector_path, coords, pixel_size, burn_value, output_format, save_path)
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

    def sliding_window(self, window_size, stride=None):
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

    def centroid_coordinates(self, vector_path, window_size, naming_field=None):

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
        half_size = window_size / 2
        
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
            geom = feature.GetGeometryRef()
            centroid = geom.Centroid()
            
            x, y = centroid.GetX(), centroid.GetY()
            
            # Get name from field or use FID
            if naming_field and field_exists:
                name = str(feature.GetField(naming_field))
            else:
                # Use FID for naming (FID is always available)
                name = f"FID_{feature.GetFID()}"
            
            coordinates.append((
                x - half_size,  # minx
                y + half_size,  # maxy
                x + half_size,  # maxx
                y - half_size   # miny
            ))
            names.append(name)
            
        ds = None
    
        return tuple(coordinates), tuple(names)

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

    def clip_raster(self, raster_path, coords, output_format="array", save_path=None):
        """
        Clip raster data for given coordinates with improved GDAL options handling.
        
        Args:
            raster_path (str): Path to raster file
            coords (tuple): (minx, maxy, maxx, miny) coordinates
            output_format (str): 'array' or 'geocoded'
            save_path (str, optional): Path to save output if desired
        
        Returns:
            numpy.ndarray or gdal.Dataset: Clipped data in requested format
        """
        # Validate inputs
        self._validate_raster_crs(raster_path)
        if not self._validate_coordinates(coords):
            raise ValueError("Coordinates outside project extent")

        # Extract coordinates
        minx, maxy, maxx, miny = coords
        
        # Open source raster
        src_ds = gdal.Open(raster_path)
        
        # Calculate dimensions
        width = int((maxx - minx) / self.pixel_size)
        height = int((maxy - miny) / self.pixel_size)
        
        # Set up warp options
        warp_options = gdal.WarpOptions(
            format='MEM' if output_format == "array" else 'GTiff',
            outputBounds=[minx, miny, maxx, maxy],
            width=width,
            height=height,
            resampleAlg=gdal.GRA_Bilinear,
            srcSRS=self.spatial_ref.ExportToWkt(),
            dstSRS=self.spatial_ref.ExportToWkt(),
            multithread=True,
            creationOptions=['COMPRESS=LZW'] if output_format == "geocoded" else None
        )
        
        # Perform warping
        if output_format == "array":
            target_ds = gdal.Warp('', src_ds, options=warp_options)
            array = target_ds.ReadAsArray()
            target_ds = None
            if save_path:
                cv2.imwrite(save_path, array)
            return array
        else:  # geocoded
            if not save_path:
                raise ValueError("save_path required for geocoded output")
            target_ds = gdal.Warp(save_path, src_ds, options=warp_options)
            return target_ds

    def clip_vector(self, vector_path, coords, pixel_size=None, burn_value=1.0,
                output_format="array", save_path=None):
        """
        Clip vector data for given coordinates with improved GDAL options handling.
        
        Args:
            vector_path (str): Path to vector file
            coords (tuple): (minx, maxy, maxx, miny) coordinates
            pixel_size (float, optional): Pixel size for rasterization
            burn_value (float): Value to burn into output raster
            output_format (str): 'array' or 'geocoded'
            save_path (str, optional): Path to save output if desired
        
        Returns:
            numpy.ndarray or gdal.Dataset: Clipped and rasterized data
        """
        # Validate inputs
        self._validate_vector_crs(vector_path)
        if not self._validate_coordinates(coords):
            raise ValueError("Coordinates outside project extent")

        # Set pixel size
        pixel_size = pixel_size or self.pixel_size
        minx, maxy, maxx, miny = coords
        
        # Calculate dimensions
        width = int((maxx - minx) / pixel_size)
        height = int((maxy - miny) / pixel_size)
        
        # Create spatial filter for clipping
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(minx, miny)
        ring.AddPoint(maxx, miny)
        ring.AddPoint(maxx, maxy)
        ring.AddPoint(minx, maxy)
        ring.AddPoint(minx, miny)
        
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        # Open and clip vector
        vector_ds = ogr.Open(vector_path)
        layer = vector_ds.GetLayer()
        layer.SetSpatialFilter(poly)
        
        # Set up rasterize options
        rasterize_options = gdal.RasterizeOptions(
            format='MEM' if output_format == "array" else 'GTiff',
            outputBounds=[minx, miny, maxx, miny],
            width=width,
            height=height,
            bands=[1],
            burnValues=[burn_value],
            allTouched=True,
            outputSRS=self.spatial_ref.ExportToWkt(),
            creationOptions=['COMPRESS=LZW'] if output_format == "geocoded" else None
        )
        
        # Perform rasterization
        if output_format == "array":
            target_ds = gdal.Rasterize('', vector_ds, options=rasterize_options)
            array = target_ds.ReadAsArray()
            target_ds = None
            if save_path:
                cv2.imwrite(save_path, array)
            return array
        else:  # geocoded
            if not save_path:
                raise ValueError("save_path required for geocoded output")
            target_ds = gdal.Rasterize(save_path, vector_ds, options=rasterize_options)
            return target_ds

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'ds'):
            self.ds = None

        gdal.PopErrorHandler()  # Restore default error handling

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
                               burn_value, output_format, output_dir, prefix, n_workers):
        """Handle parallel processing for vector clipping with progress bar."""
        with mp.Pool(n_workers, initializer=_init_gdal) as pool:
            args_list = [
                (chunk, vector_path, pixel_size, burn_value, output_format, output_dir, 
                 prefix, self.base_raster_path)
                for chunk in zip(coordinates, names)
            ]
            
            results = list(tqdm(
                pool.imap_unordered(_process_vector_chunk_helper, args_list),
                total=len(coordinates),
                desc=f"Clipping vector with {n_workers} workers",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ))
            
            # Filter out None results and collect errors
            errors = [r for r in results if r is not None]
            if errors:
                for error in errors:
                    print(error)
                raise RuntimeError("Some chunks failed to process")

    def _serial_process_vector(self, coordinates, names, vector_path, pixel_size, 
                             burn_value, output_format, output_dir, prefix):
        """Handle serial processing for vector clipping with progress bar."""
        for coords, name in tqdm(zip(coordinates, names), 
                               total=len(coordinates),
                               desc="Clipping vector",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            args = ((coords, name), vector_path, pixel_size, burn_value, output_format, 
                   output_dir, prefix, self.base_raster_path)
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
                    pixel_size=None, burn_value=1.0, output_format="geocoded", 
                    prefix="", n_workers=1):
        """Clip vector for multiple coordinates with optional parallel processing."""
        self._validate_inputs(coordinates, names)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing {len(coordinates)} vector clips...")

        try:
            if n_workers > 1:
                n_workers = self._validate_workers(n_workers)
                self._parallel_process_vector(
                    coordinates, names, vector_path, pixel_size, burn_value,
                    output_format, output_dir, prefix, n_workers
                )
            else:
                self._serial_process_vector(
                    coordinates, names, vector_path, pixel_size, burn_value,
                    output_format, output_dir, prefix
                )
            print("\nVector clipping completed!")
        except Exception as e:
            print(f"\nError during vector clipping: {str(e)}")
            raise

    def __del__(self):
        """Cleanup and restore GDAL error handling."""
        super().__del__()
        gdal.PopErrorHandler()
