"""
This module contains the functionalities to clip/crop raster dataset into cropped raster dataset and 
clip/crop the vector dataset into rasterized raster dataset by given coordinates of the Area of Interest.

"""



from osgeo import gdal, osr, ogr
import numpy as np
import cv2
import os


class RasterClipper:
    """
    A class to handle raster clipping operations using GDAL.
    Supports both geocoded and non-geocoded outputs, and both geographic and pixel coordinates.


    Assumptions:
    -----------
        - This class only works on the single raster image and not on multiple channels/layers.
    """
    
    def __init__(self, raster_path):
        """
        Initialize the RasterClipper with a raster file path.
        
        Args:
            raster_path (str): Path to the input raster file
        """
        self.raster_path = raster_path
        self.dataset = None
        self.load_dataset()
        
    def load_dataset(self):
        """Load the GDAL dataset and get basic information."""
        try:
            self.dataset = gdal.Open(self.raster_path)
            if self.dataset is None:
                raise ValueError("Could not open the raster dataset")
                
            self.geotransform = self.dataset.GetGeoTransform()
            self.projection = self.dataset.GetProjection()
            self.bands = self.dataset.RasterCount
            self.width = self.dataset.RasterXSize
            self.height = self.dataset.RasterYSize
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def geo_to_pixel(self, x, y):
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            x (float): X coordinate (longitude or easting)
            y (float): Y coordinate (latitude or northing)
            
        Returns:
            tuple: (pixel_x, pixel_y)
        """
        pixel_x = int((x - self.geotransform[0]) / self.geotransform[1])
        pixel_y = int((y - self.geotransform[3]) / self.geotransform[5])
        return pixel_x, pixel_y

    def pixel_to_geo(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel_x (int): X pixel coordinate
            pixel_y (int): Y pixel coordinate
            
        Returns:
            tuple: (geo_x, geo_y)
        """
        geo_x = self.geotransform[0] + pixel_x * self.geotransform[1]
        geo_y = self.geotransform[3] + pixel_y * self.geotransform[5]
        return geo_x, geo_y

    def clip_by_pixels(self, pixel_bbox, output_format="array", save_output=False, output_path="clipped.tif"):
        """
        Clip the raster using pixel coordinates.
        
        Args:
            pixel_bbox (tuple): Bounding box in pixel coordinates (x1, y1, x2, y2)
            output_format (str): Format of output - either "array" or "geocoded"
            save_output (bool): Whether to save the output to disk
            output_path (str): Path for saving output if save_output is True
            
        Returns:
            numpy.ndarray or gdal.Dataset: 
            - If output_format="array": returns numpy array
            - If output_format="geocoded": returns GDAL dataset
        """
        if output_format not in ["array", "geocoded"]:
            raise ValueError('output_format must be either "array" or "geocoded"')

        if output_format == "array":
            clipped_array = self._clip_array_by_pixels(pixel_bbox)
            if save_output:
                cv2.imwrite(output_path, clipped_array)   # Assumption is that the clipped_array is a single channel one.
            return clipped_array
        else:  # geocoded
            dataset = self._clip_geocoded_by_pixels(pixel_bbox)
            if save_output:
                driver = gdal.GetDriverByName('GTiff')
                driver.CreateCopy(output_path, dataset)
            return dataset

    def clip_by_coords(self, geo_bbox, output_format="array", save_output=False, output_path="clipped.tif"):
        """
        Clip the raster using geographic coordinates.
        
        Args:
            geo_bbox (tuple): Bounding box in geographic coordinates (minx, maxy, maxx, miny)
            output_format (str): Format of output - either "array" or "geocoded"
            save_output (bool): Whether to save the output to disk
            output_path (str): Path for saving output if save_output is True
            
        Returns:
            numpy.ndarray or gdal.Dataset: 
            - If output_format="array": returns numpy array
            - If output_format="geocoded": returns GDAL dataset
        """
        minx, miny, maxx, maxy = geo_bbox
        x1, y1 = self.geo_to_pixel(minx, maxy)  # Note: maxy for y1
        x2, y2 = self.geo_to_pixel(maxx, miny)  # Note: miny for y2
        pixel_bbox = (x1, y1, x2, y2)

        return self.clip_by_pixels(pixel_bbox, output_format, save_output, output_path)

    def clip_by_center(self, center_geo_x, center_geo_y, bbox_width, bbox_height,
                           output_format="array", save_output=False, output_path="clipped.tif"):
        """
        Clip the raster using a center point and dimensions.
        
        Args:
            center_geo_x (float): X geographic coordinate of center point
            center_geo_y (float): Y geographic coordinate of center point
            bbox_width (int): Width of the clipping box in pixels
            bbox_height (int): Height of the clipping box in pixels
            output_format (str): Format of output - either "array" or "geocoded"
            save_output (bool): Whether to save the output to disk
            output_path (str): Path for saving output if save_output is True
            
        Returns:
            numpy.ndarray or gdal.Dataset: 
            - If output_format="array": returns numpy array
            - If output_format="geocoded": returns GDAL dataset
        """
        center_pixel_x, center_pixel_y = self.geo_to_pixel(center_geo_x, center_geo_y)
        half_width = bbox_width // 2
        half_height = bbox_height // 2
        
        pixel_bbox = (
            center_pixel_x - half_width,
            center_pixel_y - half_height,
            center_pixel_x - half_width + bbox_width,
            center_pixel_y - half_height + bbox_height
        )
        
        return self.clip_by_pixels(pixel_bbox, output_format, save_output, output_path)

    def _clip_array_by_pixels(self, pixel_bbox):
        """
        Private method to clip raster to numpy array.
        
        Args:
            pixel_bbox (tuple): Bounding box in pixel coordinates (x1, y1, x2, y2)
            
        Returns:
            numpy.ndarray: Clipped raster data as numpy array

        Assumption:
            It is assumed that the raster dataset has only single raster channel/layer.
        """
        try:
            x1, y1, x2, y2 = pixel_bbox
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, self.width))
            x2 = max(0, min(x2, self.width))
            y1 = max(0, min(y1, self.height))
            y2 = max(0, min(y2, self.height))
            
            # Calculate dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Create output array
            if self.bands == 1:
                clipped_array = self.dataset.GetRasterBand(1).ReadAsArray(x1, y1, width, height)
            else:
                clipped_array = np.zeros((self.bands, height, width))
                for band in range(1, self.bands + 1):
                    clipped_array[band-1] = self.dataset.GetRasterBand(band).ReadAsArray(x1, y1, width, height)
            
            return clipped_array
            
        except Exception as e:
            raise Exception(f"Error clipping raster to array: {str(e)}")

    def _clip_geocoded_by_pixels(self, pixel_bbox):
        """
        Private method to clip raster and return as GDAL dataset in memory.
        
        Args:
            pixel_bbox (tuple): Bounding box in pixel coordinates (x1, y1, x2, y2)
            
        Returns:
            gdal.Dataset: Clipped raster as a GDAL dataset
        """
        try:
            x1, y1, x2, y2 = pixel_bbox
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, self.width))
            x2 = max(0, min(x2, self.width))
            y1 = max(0, min(y1, self.height))
            y2 = max(0, min(y2, self.height))
            
            # Calculate dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Create output dataset in memory
            mem_driver = gdal.GetDriverByName('MEM')
            output_dataset = mem_driver.Create(
                '', 
                width, 
                height, 
                self.bands, 
                self.dataset.GetRasterBand(1).DataType
            )
            
            # Set spatial reference
            output_dataset.SetProjection(self.projection)
            
            # Calculate new geotransform
            new_geotransform = list(self.geotransform)
            new_geotransform[0] = self.geotransform[0] + x1 * self.geotransform[1]
            new_geotransform[3] = self.geotransform[3] + y1 * self.geotransform[5]
            output_dataset.SetGeoTransform(new_geotransform)
            
            # Copy data
            for band in range(1, self.bands + 1):
                clipped_array = self.dataset.GetRasterBand(band).ReadAsArray(x1, y1, width, height)
                output_dataset.GetRasterBand(band).WriteArray(clipped_array)
            
            return output_dataset
            
        except Exception as e:
            raise Exception(f"Error clipping raster: {str(e)}")

    def __del__(self):
        """Cleanup GDAL dataset."""
        if self.dataset is not None:
            self.dataset = None




class VectorClipper:
    """
    A class to handle vector clipping operations using GDAL/OGR.
    Built step by step with memory efficiency in mind.

    Assumption:
        The main assumption in this class is that there is only one class of objects. 
        So, all the object pixel values will be the same in the rasterized clipped vector output.
    """
    
    def __init__(self, vector_path):
        """Initialize with vector file path."""
        if not isinstance(vector_path, str):
            raise TypeError("vector_path must be a string")
            
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Vector file not found: {vector_path}")
            
        self.vector_path = vector_path
        self.dataset = None
        self.layer = None
        self.spatial_ref = None
        self.load_dataset()

    def load_dataset(self):
        """Load the OGR dataset and get basic information."""
        try:
            self.dataset = ogr.Open(self.vector_path)
            if self.dataset is None:
                raise ValueError(f"Could not open the vector dataset: {self.vector_path}")
            
            self.layer = self.dataset.GetLayer(0)
            if self.layer is None:
                raise ValueError("Could not get the vector layer")
            
            self.spatial_ref = self.layer.GetSpatialRef()
            if self.spatial_ref is None:
                raise ValueError("Could not get spatial reference from layer")
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def _create_bbox_geometry(self, geo_bbox):
        """Create an OGR polygon geometry from a bounding box."""
        try:
            if len(geo_bbox) != 4:
                raise ValueError("Bounding box must have exactly 4 coordinates")
                
            minx, maxy, maxx, miny = geo_bbox
            
            if minx >= maxx or miny >= maxy:
                raise ValueError("Invalid bbox coordinates")
            
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(minx, miny)
            ring.AddPoint(maxx, miny)
            ring.AddPoint(maxx, maxy)
            ring.AddPoint(minx, maxy)
            ring.AddPoint(minx, miny)
            
            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)
            
            return polygon
            
        except Exception as e:
            raise Exception(f"Error creating bbox geometry: {str(e)}")

    def calculate_bbox(self, center_geo_x, center_geo_y, width, height):
        """Calculate bounding box coordinates from center point and dimensions."""
        try:
            half_width = width / 2
            half_height = height / 2
            
            minx = center_geo_x - half_width
            maxx = center_geo_x + half_width
            miny = center_geo_y - half_height
            maxy = center_geo_y + half_height
            
            return (minx, maxy, maxx, miny)
            
        except Exception as e:
            raise Exception(f"Error calculating bbox: {str(e)}")

    def get_feature_count(self, geometry):
        """Get the number of features that intersect with the geometry."""
        try:
            self.layer.SetSpatialFilter(geometry)
            count = self.layer.GetFeatureCount()
            self.layer.SetSpatialFilter(None)
            return count
        except Exception as e:
            raise Exception(f"Error getting feature count: {str(e)}")
    
    def _create_output_layer(self, output_path, geom_type=ogr.wkbPolygon):
        """Create a new shapefile for output."""
        try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            if os.path.exists(output_path):
                driver.DeleteDataSource(output_path)
            
            out_ds = driver.CreateDataSource(output_path)
            out_layer = out_ds.CreateLayer('clipped', self.spatial_ref, geom_type)
            
            # Copy field definitions
            layer_defn = self.layer.GetLayerDefn()
            for i in range(layer_defn.GetFieldCount()):
                field_defn = layer_defn.GetFieldDefn(i)
                out_layer.CreateField(field_defn)
            
            return out_ds, out_layer
            
        except Exception as e:
            raise Exception(f"Error creating output layer: {str(e)}")

    def _calculate_raster_dimensions(self, bbox, pixel_size):
        """
        Calculate raster dimensions based on bbox and pixel size.
        
        Args:
            bbox (tuple): (minx, maxy, maxx, miny)
            pixel_size (float): Size of pixels in map units
            
        Returns:
            tuple: (width, height, geotransform)
        """
        try:
            minx, maxy, maxx, miny = bbox
            
            # Calculate dimensions
            width = int((maxx - minx) / pixel_size)
            height = int((maxy - miny) / pixel_size)
            
            # Create geotransform
            geotransform = (minx, pixel_size, 0, maxy, 0, -pixel_size)
            
            return width, height, geotransform
            
        except Exception as e:
            raise Exception(f"Error calculating raster dimensions: {str(e)}")

    def _save_output(self, data, output_format, output_path, geotransform=None, save_output=False):
        """
        Save output data in the appropriate format.
        
        Args:
            data (numpy.ndarray): Data to save
            output_format (str): Either 'array' or 'geocoded'
            output_path (str): Path to save the output
            geotransform (tuple): GDAL geotransform for geocoded output
            save_output (bool): Whether to save the output
        """
        if save_output and output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_format == 'array':
                # Save as simple binary image using OpenCV
                binary_array = (data > 0).astype(np.uint8) * 255
                cv2.imwrite(output_path, binary_array)
            else:  # 'geocoded'
                mem_driver = gdal.GetDriverByName('MEM')
                dataset = mem_driver.Create('', data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
                dataset.SetGeoTransform(geotransform)
                dataset.SetProjection(self.spatial_ref.ExportToWkt())
                dataset.GetRasterBand(1).WriteArray(data)
                
                driver = gdal.GetDriverByName('GTiff')
                output_ds = driver.CreateCopy(output_path, dataset)
                output_ds = None
                dataset = None

    def rasterize_clip(self, bbox, pixel_size, burn_value=1, output_format="array", output_path=None):
        """
        Clip features to bbox and rasterize the result.
        
        Args:
            bbox (tuple): Bounding box coordinates (minx, maxy, maxx, miny)
            pixel_size (float): Size of pixels in map units
            burn_value (float): Value to burn into raster (default: 1)
            output_format (str): Either 'array' or 'geocoded'
            output_path (str): Optional path to save the raster
            
        Returns:
            numpy.ndarray: Rasterized data as numpy array
        """
        try:
            # Create temporary shapefile for clipped features
            temp_dir = os.path.dirname(output_path) if output_path else 'temp_dir'
            os.makedirs(temp_dir, exist_ok=True)
            temp_vector = os.path.join(temp_dir, 'temp_clip.shp')
            
            # Clip features
            num_features = self.clip_features_to_bbox(bbox, temp_vector)
            if num_features == 0:
                return np.zeros((1, 1), dtype=np.float32)
            
            # Calculate raster dimensions
            width, height, geotransform = self._calculate_raster_dimensions(bbox, pixel_size)
            
            # Create raster in memory
            mem_driver = gdal.GetDriverByName('MEM')
            raster_ds = mem_driver.Create('', width, height, 1, gdal.GDT_Float32)
            raster_ds.SetGeoTransform(geotransform)
            raster_ds.SetProjection(self.spatial_ref.ExportToWkt())
            
            # Initialize with zeros
            raster_band = raster_ds.GetRasterBand(1)
            raster_band.Fill(0)
            
            # Open clipped features
            vector_ds = ogr.Open(temp_vector)
            vector_layer = vector_ds.GetLayer()
            
            # Rasterize
            gdal.RasterizeLayer(raster_ds, [1], vector_layer, burn_values=[burn_value])
            
            # Get data as array
            array = raster_band.ReadAsArray()
            
            # Save based on format if path provided
            if output_path:
                self._save_output(array, output_format, output_path, geotransform, True)
            
            # Cleanup
            raster_ds = None
            vector_ds = None
            
            # Remove temporary files
            if os.path.exists(temp_vector):
                driver = ogr.GetDriverByName('ESRI Shapefile')
                driver.DeleteDataSource(temp_vector)
            
            return array
            
        except Exception as e:
            raise Exception(f"Error in rasterize_clip: {str(e)}")
        
    def clip_features_to_bbox(self, bbox, output_path):
        """
        Clip features to bbox and save to new shapefile.
        
        Args:
            bbox (tuple): Bounding box coordinates (minx, maxy, maxx, miny)
            output_path (str): Path for output shapefile
            
        Returns:
            int: Number of features clipped
        """
        try:
            # Create clip geometry
            clip_geom = self._create_bbox_geometry(bbox)
            
            # Check feature count
            feature_count = self.get_feature_count(clip_geom)
            if feature_count == 0:
                print("No features found in clip area")
                return 0
            
            # Create output
            out_ds, out_layer = self._create_output_layer(output_path)
            
            # Process features one at a time
            self.layer.SetSpatialFilter(clip_geom)
            features_processed = 0
            
            for feature in self.layer:
                geom = feature.GetGeometryRef()
                if geom is not None:
                    # Create new feature
                    out_feature = ogr.Feature(out_layer.GetLayerDefn())
                    out_feature.SetGeometry(geom.Clone())
                    
                    # Copy attributes
                    for i in range(feature.GetFieldCount()):
                        out_feature.SetField(i, feature.GetField(i))
                    
                    # Add to layer
                    out_layer.CreateFeature(out_feature)
                    features_processed += 1
                    
                    # Cleanup
                    out_feature = None
                    geom = None
                feature = None
            
            # Reset spatial filter
            self.layer.SetSpatialFilter(None)
            
            # Cleanup
            out_ds = None
            
            return features_processed
            
        except Exception as e:
            raise Exception(f"Error in clip_features_to_bbox: {str(e)}")

    def clip_by_center(self, center_geo_x, center_geo_y, bbox_width, bbox_height,
                      pixel_size, burn_value=1, output_format="array", 
                      save_output=False, output_path=None):
        """
        Clip and rasterize vector using a center point and pixel dimensions.
        
        Args:
            center_geo_x (float): X coordinate of center point
            center_geo_y (float): Y coordinate of center point
            bbox_width (int):Width of the clipping box in pixels
            bbox_height (int): Height of the clipping box in pixels
            pixel_size (float): Size of pixels in map units
            burn_value (float): Value to burn into raster (default: 1)
            output_format (str): Either 'array' or 'geocoded'
            save_output (bool): Whether to save to disk
            output_path (str): Path for output file if saving
            
        Returns:
            numpy.ndarray or gdal.Dataset: Based on output_format
        """
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [center_geo_x, center_geo_y]):
                raise ValueError("Center coordinates must be numeric")
                
            if not all(isinstance(x, int) and x > 0 for x in [bbox_width, bbox_height]):
                raise ValueError("Pixel dimensions must be positive integers")
                
            if not isinstance(pixel_size, (int, float)) or pixel_size <= 0:
                raise ValueError("Pixel size must be a positive number")

            # Calculate real-world width and height
            width_map_units = bbox_width * pixel_size
            height_map_units = bbox_height * pixel_size

            # Calculate bbox
            bbox = self.calculate_bbox(
                center_geo_x, 
                center_geo_y, 
                width_map_units, 
                height_map_units
            )

            # Calculate geotransform for final output
            minx = center_geo_x - (bbox_width * pixel_size) / 2
            maxy = center_geo_y + (bbox_height * pixel_size) / 2
            geotransform = (minx, pixel_size, 0, maxy, 0, -pixel_size)

            # Perform rasterization
            result = self.rasterize_clip(
                bbox=bbox,
                pixel_size=pixel_size,
                burn_value=burn_value,
                output_format=output_format,
                output_path=output_path if save_output else None
            )

            if output_format == 'array':
                return result
            else:  # 'geocoded'
                # Create geocoded dataset
                mem_driver = gdal.GetDriverByName('MEM')
                out_ds = mem_driver.Create('', bbox_width, bbox_height, 1, gdal.GDT_Float32)
                out_ds.SetProjection(self.spatial_ref.ExportToWkt())
                out_ds.SetGeoTransform(geotransform)
                out_ds.GetRasterBand(1).WriteArray(result)
                return out_ds
                
        except Exception as e:
            raise Exception(f"Error in clip_by_center: {str(e)}")

    def clip_by_coords(self, geo_bbox, pixel_size, burn_value=1, 
                      output_format="array", save_output=False, output_path=None):
        """
        Clip and rasterize vector using coordinate bounds.
        
        Args:
            geo_bbox (tuple): Bounding box coordinates (minx, maxy, maxx, miny)
            pixel_size (float): Size of pixels in map units
            burn_value (float): Value to burn into raster (default: 1)
            output_format (str): Either 'array' or 'geocoded'
            save_output (bool): Whether to save to disk
            output_path (str): Path for output file if saving
            
        Returns:
            numpy.ndarray or gdal.Dataset: Based on output_format
        """
        try:
            if len(geo_bbox) != 4:
                raise ValueError("geo_bbox must have exactly 4 coordinates (minx, maxy, maxx, miny)")
                
            minx, maxy, maxx, miny = geo_bbox
            
            if minx >= maxx or miny >= maxy:
                raise ValueError("Invalid bbox coordinates: minx must be less than maxx and miny must be less than maxy")

            # Calculate pixel dimensions
            pixel_width = int((maxx - minx) / pixel_size)
            pixel_height = int((maxy - miny) / pixel_size)

            # Create geotransform
            geotransform = (minx, pixel_size, 0, maxy, 0, -pixel_size)

            # Perform rasterization
            result = self.rasterize_clip(
                bbox=geo_bbox,
                pixel_size=pixel_size,
                burn_value=burn_value,
                output_format=output_format,
                output_path=output_path if save_output else None
            )

            if output_format == 'array':
                return result
            else:  # 'geocoded'
                # Create geocoded dataset
                mem_driver = gdal.GetDriverByName('MEM')
                out_ds = mem_driver.Create('', pixel_width, pixel_height, 1, gdal.GDT_Float32)
                out_ds.SetProjection(self.spatial_ref.ExportToWkt())
                out_ds.SetGeoTransform(geotransform)
                out_ds.GetRasterBand(1).WriteArray(result)
                return out_ds
                
        except Exception as e:
            raise Exception(f"Error in clip_by_coords: {str(e)}")
        
    def clip_by_geom(self, clip_geometry, pixel_size, burn_value=1,
                    output_format="array", save_output=False, output_path=None):
        """
        Clip and rasterize vector using an OGR geometry.
        
        Args:
            clip_geometry (ogr.Geometry): OGR geometry to use as clip boundary
            pixel_size (float): Size of pixels in map units
            burn_value (float): Value to burn into raster (default: 1)
            output_format (str): Either 'array' or 'geocoded'
            save_output (bool): Whether to save to disk
            output_path (str): Path for output file if saving
            
        Returns:
            numpy.ndarray or gdal.Dataset: Based on output_format
        """
        try:
            if not clip_geometry:
                raise ValueError("clip_geometry cannot be None")
                
            # Get geometry envelope
            minx, maxx, miny, maxy = clip_geometry.GetEnvelope()
            geo_bbox = (minx, maxy, maxx, miny)  # Convert to our format

            # Calculate pixel dimensions
            pixel_width = int((maxx - minx) / pixel_size)
            pixel_height = int((maxy - miny) / pixel_size)

            # Create geotransform
            geotransform = (minx, pixel_size, 0, maxy, 0, -pixel_size)

            # Create temporary memory datasource for the clip geometry
            mem_driver = ogr.GetDriverByName('Memory')
            temp_ds = mem_driver.CreateDataSource('')
            temp_layer = temp_ds.CreateLayer('clip', self.spatial_ref, clip_geometry.GetGeometryType())
            
            # Create feature with the clip geometry
            feat_defn = temp_layer.GetLayerDefn()
            feat = ogr.Feature(feat_defn)
            feat.SetGeometry(clip_geometry)
            temp_layer.CreateFeature(feat)

            # Create output raster
            mem_driver = gdal.GetDriverByName('MEM')
            raster_ds = mem_driver.Create('', pixel_width, pixel_height, 1, gdal.GDT_Float32)
            raster_ds.SetGeoTransform(geotransform)
            raster_ds.SetProjection(self.spatial_ref.ExportToWkt())
            raster_ds.GetRasterBand(1).Fill(0)

            # Set spatial filter and rasterize
            self.layer.SetSpatialFilter(clip_geometry)
            gdal.RasterizeLayer(raster_ds, [1], self.layer, burn_values=[burn_value])
            self.layer.SetSpatialFilter(None)

            # Get result as array
            result = raster_ds.GetRasterBand(1).ReadAsArray()

            if save_output and output_path:
                self._save_output(result, output_format, output_path, geotransform, True)

            if output_format == 'array':
                raster_ds = None
                temp_ds = None
                return result
            else:  # 'geocoded'
                temp_ds = None
                return raster_ds

        except Exception as e:
            raise Exception(f"Error in clip_by_geom: {str(e)}")

    def __del__(self):
        """Cleanup when object is deleted."""
        if self.dataset is not None:
            self.dataset = None