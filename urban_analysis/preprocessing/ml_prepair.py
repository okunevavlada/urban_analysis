"""
Модуль для подготовки параметров для модели машинного обучения 
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.affinity import rotate
from tqdm import tqdm
import os

def calculate_building_parameters(
    buildings_geojson: str,
    width_csv: str,
    lcz_csv: str,
    dem_csv: str,
    output_csv: str,
    crs_code: int = 32645
) -> pd.DataFrame:
    
    """
    Расчет параметров зданий и объединение с существующими данными
    
    Args:
        buildings_geojson: путь к исходному GeoJSON со зданиями
        width_csv: путь к CSV с шириной (должен содержать колонку width_med)
        lcz_csv: путь к CSV с LCZ (должен содержать колонку majority_100m)
        dem_csv: путь к CSV с высотами (должен содержать колонку max_value)
        output_csv: путь для сохранения результата
        crs_code: код метрической проекции (по умолчанию 32645 - UTM 45N)
    
    Returns:
        pd.DataFrame: объединенный DataFrame со всеми параметрами
    """
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from shapely import wkt
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.affinity import rotate
    from tqdm import tqdm
    import os

    print("\n1. Загрузка зданий")
    gdf = gpd.read_file(buildings_geojson)
    gdf = gdf.to_crs(crs_code)

    print("\n2. Загрузка рассчитанных параметров")
    
    df_width = pd.read_csv(width_csv)
    df_lcz = pd.read_csv(lcz_csv)
    df_dem = pd.read_csv(dem_csv)

    print("\n3. Объединение параметров")

    gdf['id'] = gdf['id'].astype(str)
    df_dem['id'] = df_dem['id'].astype(str)
    df_width['id'] = df_width['id'].astype(str)
    df_lcz['id'] = df_lcz['id'].astype(str)

    gdf = gdf.merge(df_dem[['id', 'max_value']], on='id',how='left')
    gdf.rename(columns={'max_value': 'min_dem'}, inplace=True)
    gdf = gdf.merge(df_width[['id', 'width_med']],on='id',how='left')
    gdf = gdf.merge(df_lcz[['id', 'majority_100m']],on='id',how='left')

    
    print("\n4. Расчет геометрических параметров")
    gdf['area'] = gdf.geometry.area

    def minimum_rotated_bbox(geometry):
        convex_hull = geometry.convex_hull
        min_area = float('inf')
        best_bounds = None
        
        for angle in range(0, 180, 5):
            rotated = rotate(convex_hull, angle, origin='centroid', use_radians=False)
            minx, miny, maxx, maxy = rotated.bounds
            width = maxx - minx
            height = maxy - miny
            area = width * height
            
            if area < min_area:
                min_area = area
                best_bounds = (width, height)
        
        if best_bounds:
            return {
                'area': min_area,
                'width': best_bounds[0],
                'length': best_bounds[1]
            }
        return None
    
    min_bboxes = []
    bbox_area_ratios = []
    bbox_width_length = []
    
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        object_area = row.area
        
        if geometry.is_empty:
            min_bboxes.append(None)
            bbox_area_ratios.append(None)
            bbox_width_length.append(None)
            continue
        
        if geometry.geom_type == 'Polygon':
            result = minimum_rotated_bbox(geometry)
        elif geometry.geom_type == 'MultiPolygon':
            merged = geometry.convex_hull
            result = minimum_rotated_bbox(merged)
        else:
            result = None
        
        if result:
            min_bboxes.append(result['area'])
            if object_area > 0:
                bbox_area_ratios.append(object_area / result['area'])
            else:
                bbox_area_ratios.append(None)
            
            width = result['width']
            length = result['length']
            if length > 0:
                bbox_width_length.append(min(width / length, length / width))
            else:
                bbox_width_length.append(None)
        else:
            min_bboxes.append(None)
            bbox_area_ratios.append(None)
            bbox_width_length.append(None)
    
    gdf['obox_ratio'] = bbox_area_ratios
    gdf['obox_hw'] = bbox_width_length

    gdf['isoquotient'] = (4 * 3.14159 * gdf['area']) / (gdf.geometry.length ** 2)

    
    def calculate_height(row):
        return row['num_floors'] * 3
    
    def get_calc_height(row):
        height = row['height']
        if pd.notna(height) and height != 0:
            return height
        else:
            return calculate_height(row)
    
    gdf['calc_height'] = gdf.apply(get_calc_height, axis=1)


    gdf['geometry_wkt'] = gdf.geometry.apply(lambda x: x.wkt)

    attrs_df = gdf.drop(columns=['geometry'])

    attrs_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Колонки: {list(attrs_df.columns)}")
    
    return attrs_df