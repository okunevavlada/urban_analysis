"""
Растеризация зданий
"""

import math
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import os
from typing import Optional


def rasterize_buildings(
    buildings_path: str,
    output_raster: str,
    resolution: float = 2.0,
    crs_code: int = 32645,
    fill_value: int = 1
) -> str:
    """
    Растеризация зданий из GeoJSON
    
    Args:
        buildings_path: путь к GeoJSON со зданиями
        output_raster: путь для сохранения растра
        resolution: разрешение выходной сетки
        crs_code: код проекции (например 32645 для UTM 45N)
        fill_value: значение заполнения для зданий
    
    Returns:
        путь к созданному растру
    """
    os.makedirs(os.path.dirname(output_raster), exist_ok=True)
    
    df = gpd.read_file(buildings_path)
    df = df.to_crs(crs_code)

    def floor_base(x, base=1):
        return base * math.floor(x / base)
    
    def ceil_base(x, base=1):
        return base * math.ceil(x / base)

    box = df.geometry.total_bounds
    west = floor_base(box[0], resolution) - resolution
    east = ceil_base(box[2], resolution)
    south = floor_base(box[1], resolution)
    north = ceil_base(box[3], resolution) + resolution
    
    nrow = int(1 + (north - south) / resolution)
    ncol = int(1 + (east - west) / resolution)

    
    transform = rasterio.transform.from_origin(west, north, resolution, resolution)

    build_raster = rasterize(
        df.geometry,
        out_shape=(nrow, ncol),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    with rasterio.open(
        output_raster,
        'w',
        driver='GTiff',
        height=build_raster.shape[0],
        width=build_raster.shape[1],
        count=1,
        dtype='uint8',
        compress='lzw',
        crs=df.crs,
        transform=transform
    ) as dst:
        dst.write(build_raster, 1)
    
    return output_raster