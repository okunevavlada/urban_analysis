"""
Функция расчета непроницаемых поверхностей 
"""

import numpy as np

def rasterize_roads(
    roads_path: str,
    building_raster_path: str,
    output_path: str,
    crs_code: int = 32645,
    super_sampling: int = 3,
    use_tqdm: bool = True
) -> np.ndarray:
    
    import os
    import ast
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from tqdm import tqdm

    """
    Растеризация дорог с буферами на основе width_value или класса
    
    Args:
        roads_path: полный путь к файлу с дорогами в формате GeoJSON
        building_raster_path: полный путь к растру зданий 
        output_path: полный путь для сохранения выходного растра
        crs_code: код проекции в EPSG (по умолчанию 32645 - UTM зона 45N)
        super_sampling: коэффициент суперсэмплинга 
        use_tqdm: показывать прогресс-бар
    
    Returns:
        np.ndarray: растр с долей дорог в процентах (0-100, тип uint8)
    """

    roads = gpd.read_file(roads_path)
    roads = roads.to_crs(epsg=crs_code)

    with rasterio.open(building_raster_path) as src:
        bounds = src.bounds
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    clip_polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs=roads.crs)
    roads_clip = gpd.clip(roads, clip_polygon)
    
    width_rules = []
    for value in roads_clip['width_value']:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = None
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                width_value = value[0].get('value')
            else:
                width_value = None
        else:
            width_value = None
        width_rules.append(width_value)
    
    roads_clip['width_value'] = width_rules
    roads_clip['width_value'] = pd.to_numeric(roads_clip['width_value'], errors='coerce')

    def calculate_buffer(row):
        width_value = row.get('width_value')
        subtype = row.get('subtype')
        road_class = row.get('class')
        
        if pd.notna(width_value):
            return float(width_value) / 2

        if road_class == 'path':
            return 0
        
        if subtype == 'road':
            if road_class in ['cycleway', 'road', 'bridleway', 'unclassified']:
                return 2.25
            elif road_class in ['living_street', 'residential', 'service']:
                return 3
            elif road_class in ['primary', 'primary_link', 'secondary', 'secondary_link',
                               'tertiary', 'tertiary_link', 'unknown']:
                return 8
            elif road_class in ['trunk', 'trunk_link', 'motorway', 'motorway_link', 'raceway']:
                return 8
            elif road_class in ['footway', 'pedestrian']:
                return 0.5
        
        return 0

    buffered_geometries = []
    for idx, row in roads_clip.iterrows():
        buffer_dist = calculate_buffer(row)
        if buffer_dist > 0:
            buffered = row.geometry.buffer(buffer_dist)
            if not buffered.is_empty:
                buffered_geometries.append(buffered)

    sub_width = width * super_sampling
    sub_height = height * super_sampling
    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
    sub_pixel_x = (xmax - xmin) / sub_width
    sub_transform = from_origin(xmin, ymax, sub_pixel_x, sub_pixel_x)
    shapes = [(geom, 1) for geom in buffered_geometries if not geom.is_empty]

    if use_tqdm:
        print(f"Растеризация ")
    
    sub_mask = rasterize(
        shapes, 
        out_shape=(sub_height, sub_width),
        transform=sub_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )

    reshaped = sub_mask.reshape(height, super_sampling, width, super_sampling)
    block_sums = reshaped.sum(axis=(1, 3))
    total_pixels = super_sampling * super_sampling
    result = (block_sums * 100 / total_pixels).astype(np.uint8)
    result = np.minimum(result, 100)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
        compress='lzw',
        BIGTIFF='YES'
    ) as dst:
        dst.write(result, 1)

    return result