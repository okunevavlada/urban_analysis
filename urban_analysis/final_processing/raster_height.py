
"""
Модуль для растеризации зданий с высотами
Создает два растра: по столбцам 'calc_height' и 'predicted'
"""

import math
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import wkt
import os
from pathlib import Path

def rasterize_both_heights(
    input_csv: str,
    output_dir: str,
    base_name: str = "height",
    resolution: float = 2.0,
    crs_code: int = 32636,
    geometry_column: str = 'geometry',
    calc_column: str = 'calc_height',
    pred_column: str = 'predicted'
) -> dict:
    """
    Растеризация зданий по двум колонкам высот
    
    Args:
        input_csv: путь к CSV файлу со зданиями
        output_dir: папка для сохранения растров
        base_name: базовое имя для выходных файлов
        resolution: разрешение выходной сетки
        crs_code: код проекции (например 32636 для UTM 36N)
        geometry_column: колонка с геометрией в WKT
        calc_column: колонка с рассчитанными высотами
        pred_column: колонка с предсказанными высотами
    
    Returns:
        словарь с путями к созданным растрам
    """

    os.makedirs(output_dir, exist_ok=True)

    output_calc = os.path.join(output_dir, "height_calc.tif")
    output_pred = os.path.join(output_dir, "height_pred.tif")

    df = pd.read_csv(input_csv)
    df['geometry'] = df[geometry_column].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = f'EPSG:{crs_code}'

    def floor_base(x, base=1):
        return base * math.floor(x / base)
    
    def ceil_base(x, base=1):
        return base * math.ceil(x / base)

    box = gdf.geometry.total_bounds
    west  = floor_base(box[0], resolution) - resolution
    east  = ceil_base(box[2], resolution)
    south = floor_base(box[1], resolution)
    north = ceil_base(box[3], resolution) + resolution
    
    nrow = int(1 + (north - south) / resolution)
    ncol = int(1 + (east - west) / resolution)
    
    transform = rasterio.transform.from_origin(west, north, resolution, resolution)
    
    def rasterize_column(column_name, output_path):
        
        gdf_valid = gdf[gdf[column_name].notna()].copy()

        height_raster = rasterize(
            [(geom, value) for geom, value in zip(gdf_valid.geometry, gdf_valid[column_name])],
            out_shape=(nrow, ncol),
            transform=transform,
            fill=0,
            dtype='uint16'
        )

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height_raster.shape[0],
            width=height_raster.shape[1],
            count=1,
            compress='lzw',
            BIGTIFF='YES',
            dtype='uint16',
            crs=gdf.crs,
            transform=transform
        ) as dst:
            dst.write(height_raster, 1)
        
        return output_path
    
    result_calc = rasterize_column(calc_column, output_calc)
    result_pred = rasterize_column(pred_column, output_pred)

    results = {}
    if result_calc:
        results['calc_raster'] = result_calc
    if result_pred:
        results['predicted_raster'] = result_pred
    
    return results

