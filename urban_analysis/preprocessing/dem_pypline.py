"""
Обработка ArcticDEM и извлечение высот зданий
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import glob
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from rasterstats import zonal_stats
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union 

# ОБРАБОТКА ARCTICDEM

def process_arcticdem(
    dem_tiles_pattern: str,
    roads_path: str,
    output_dir: str,
    temp_dir: str,
    buffer_degrees: float = 0.005,
    nodata_value: float = -9999.0,
    chunk_size_interp: int = 500,
    chunk_size_io: int = 1000,
    use_tqdm: bool = True
) -> dict:
    """
    Обработка ArcticDEM (исходная версия, но с контролем памяти)
    """
    import shutil
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    output_merged_path = os.path.join(output_dir, "dem_merged.tif")
    
    tile_paths = glob.glob(dem_tiles_pattern)
    if not tile_paths:
        raise FileNotFoundError(f"Тайлы не найдены: {dem_tiles_pattern}")
    
    print(f"   Найдено тайлов: {len(tile_paths)}")

    roads = gpd.read_file(roads_path)

    bounds = roads.total_bounds
    bbox = [
        bounds[0] - buffer_degrees,
        bounds[1] - buffer_degrees,
        bounds[2] + buffer_degrees,
        bounds[3] + buffer_degrees
    ]
    
    bbox_geom = box(*bbox)
    target_crs = 'EPSG:4326'
    
    temp_files = []
    
    iterator = enumerate(tile_paths)
    if use_tqdm:
        iterator = tqdm(list(iterator), desc="   Обработка тайлов")
    
    for i, tile_path in iterator:
        with rasterio.open(tile_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            temp_path = os.path.join(temp_dir, f'tile_{i}_wgs84.tif')
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'compress': 'lzw',
                'BIGTIFF': 'YES',
                'dtype': 'float32'  # ← ЯВНО УКАЗЫВАЕМ float32!
            })
            
            with rasterio.open(temp_path, 'w', **kwargs) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            
            with rasterio.open(temp_path) as src_proj:
                out_image, out_transform = mask(
                    src_proj, [bbox_geom],
                    crop=True,
                    nodata=nodata_value,
                    filled=True
                )
            
            os.remove(temp_path)
            
            clipped_path = os.path.join(temp_dir, f'tile_{i}_clipped.tif')
            with rasterio.open(
                clipped_path, 'w',
                driver='GTiff',
                height=out_image.shape[1],
                width=out_image.shape[2],
                count=1,
                dtype='float32',  # ← float32!
                crs=target_crs,
                transform=out_transform,
                compress='lzw',
                nodata=nodata_value,
                BIGTIFF='YES'
            ) as dst:
                dst.write(out_image[0].astype('float32'), 1)
            
            temp_files.append(clipped_path)
    
    if not temp_files:
        return None
    
    src_files = [rasterio.open(f) for f in temp_files]
    mosaic, out_transform = merge(src_files)
    
    for src in src_files:
        src.close()
    
    with rasterio.open(
        output_merged_path, 'w',
        driver='GTiff',
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype='float32',  # ← float32!
        crs=target_crs,
        transform=out_transform,
        compress='lzw',
        nodata=nodata_value,
        BIGTIFF='YES'
    ) as dst:
        dst.write(mosaic[0].astype('float32'), 1)
    
    for temp_path in temp_files:
        try:
            os.remove(temp_path)
        except:
            pass
    
    with rasterio.open(output_merged_path) as src:
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata
        bounds = src.bounds
        original_width = src.width
        original_height = src.height
    
    x_coords = transform.c + np.arange(original_width) * transform.a
    y_coords = transform.f + np.arange(original_height) * transform.e

    roads = gpd.read_file(roads_path)
    if raster_crs is not None and roads.crs != raster_crs:
        roads = roads.to_crs(raster_crs)
    
    all_lines = []
    for geom in roads.geometry:
        if geom.geom_type == 'LineString':
            all_lines.append(geom)
        elif geom.geom_type == 'MultiLineString':
            all_lines.extend(list(geom.geoms))
        

    lines = all_lines
    all_points_set = set()

    iterator = range(len(lines))
    if use_tqdm:
        iterator = tqdm(iterator, desc="   Пересечения")
    
    for i in iterator:
        for j in range(i+1, len(lines)):
            if lines[i].intersects(lines[j]):
                intersection = lines[i].intersection(lines[j])
                if intersection.geom_type == 'Point':
                    all_points_set.add((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        all_points_set.add((point.x, point.y))
    
    iterator = lines
    if use_tqdm:
        iterator = tqdm(iterator, desc="   Конечные точки")
    
    for line in iterator:
        if line.geom_type == 'LineString':
            coords = list(line.coords)
            if len(coords) >= 2:
                all_points_set.add(coords[0])
                all_points_set.add(coords[-1])
    
    all_points = list(all_points_set)
    
    valid_points = []
    for point in all_points:
        if (bounds.left <= point[0] <= bounds.right and 
            bounds.bottom <= point[1] <= bounds.top):
            valid_points.append(point)

    points_with_height = []
    
    with rasterio.open(output_merged_path) as src:
        iterator = valid_points
        if use_tqdm:
            iterator = tqdm(iterator, desc="   Извлечение высот")
        
        for point in iterator:
            col = int((point[0] - transform.c) / transform.a)
            row = int((point[1] - transform.f) / transform.e)
            
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1, window=((row, row+1), (col, col+1)))[0,0]
                
                if nodata is None or value != nodata:
                    points_with_height.append([point[0], point[1], value])
    
    points = np.array(points_with_height)
    
    if len(points) < 3:
        print("Недостаточно точек для TIN (менее 3)")
        tin_raster_path = None
        difference_path = None
    else:
        tri = Delaunay(points[:, :2])
        
        interpolator = LinearNDInterpolator(tri, points[:, 2])
        tin_raster_path = os.path.join(temp_dir, "tin_raster.tif")
        
        profile_tin = {
            'driver': 'GTiff',
            'height': original_height,
            'width': original_width,
            'count': 1,
            'dtype': np.float32,
            'crs': raster_crs,
            'transform': transform,
            'nodata': nodata_value,
            'compress': 'lzw',
            'BIGTIFF': 'YES' 
        }
        
        x_coords = transform.c + np.arange(original_width) * transform.a
        y_coords = transform.f + np.arange(original_height) * transform.e
        
        with rasterio.open(tin_raster_path, 'w', **profile_tin) as dst:
            block_size = chunk_size_interp
            rows_range = range(0, original_height, block_size)
            if use_tqdm:
                rows_range = tqdm(rows_range, desc="   Интерполяция")
            
            for row_start in rows_range:
                row_end = min(row_start + block_size, original_height)
                
                for col_start in range(0, original_width, block_size):
                    col_end = min(col_start + block_size, original_width)
                    
                    block_x, block_y = np.meshgrid(
                        x_coords[col_start:col_end],
                        y_coords[row_start:row_end]
                    )
                    
                    block_values = interpolator(block_x, block_y)
                    block_values = np.where(np.isnan(block_values), nodata_value, block_values)
                    
                    dst.write(
                        block_values.astype(np.float32),
                        1,
                        window=((row_start, row_end), (col_start, col_end))
                    )
        
        difference_path = os.path.join(output_dir, "difference.tif")
        
        with rasterio.open(output_merged_path) as src1, rasterio.open(tin_raster_path) as src2:
            
            overlap = [
                max(src1.bounds.left, src2.bounds.left),
                max(src1.bounds.bottom, src2.bounds.bottom),
                min(src1.bounds.right, src2.bounds.right),
                min(src1.bounds.top, src2.bounds.top)
            ]

            if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
                print("Растры не перекрываются")
                difference_path = None
            else:
                col_min1 = int((overlap[0] - src1.transform.c) / src1.transform.a)
                col_max1 = int((overlap[2] - src1.transform.c) / src1.transform.a) + 1
                row_min1 = int((src1.transform.f - overlap[3]) / (-src1.transform.e))
                row_max1 = int((src1.transform.f - overlap[1]) / (-src1.transform.e)) + 1
                
                col_min2 = int((overlap[0] - src2.transform.c) / src2.transform.a)
                col_max2 = int((overlap[2] - src2.transform.c) / src2.transform.a) + 1
                row_min2 = int((src2.transform.f - overlap[3]) / (-src2.transform.e))
                row_max2 = int((src2.transform.f - overlap[1]) / (-src2.transform.e)) + 1
                
                width = min(col_max1 - col_min1, col_max2 - col_min2)
                height = min(row_max1 - row_min1, row_max2 - row_min2)
                
                out_transform = rasterio.Affine(
                    src1.transform.a, 0, overlap[0],
                    0, src1.transform.e, overlap[3]
                )
                
                profile_diff = src1.profile.copy()
                profile_diff.update({
                    'height': height,
                    'width': width,
                    'transform': out_transform,
                    'dtype': 'float32',
                    'nodata': nodata_value,
                    'compress': 'lzw',
                    'BIGTIFF': 'YES' 
                })
                
                with rasterio.open(difference_path, 'w', **profile_diff) as dst:
                    chunk_size = chunk_size_io
                    rows_range = range(0, height, chunk_size)
                    if use_tqdm:
                        rows_range = tqdm(rows_range, desc="   Вычитание")
                    
                    for row_start in rows_range:
                        row_end = min(row_start + chunk_size, height)
                        
                        window1 = ((row_min1 + row_start, row_min1 + row_end), 
                                  (col_min1, col_min1 + width))
                        window2 = ((row_min2 + row_start, row_min2 + row_end), 
                                  (col_min2, col_min2 + width))
                        
                        chunk1 = src1.read(1, window=window1)
                        chunk2 = src2.read(1, window=window2)
                        
                        if src1.nodata is not None:
                            valid1 = (chunk1 != src1.nodata) & (~np.isnan(chunk1))
                        else:
                            valid1 = ~np.isnan(chunk1)
                            
                        if src2.nodata is not None:
                            valid2 = (chunk2 != src2.nodata) & (~np.isnan(chunk2))
                        else:
                            valid2 = ~np.isnan(chunk2)
                        
                        mask_both = valid1 & valid2
                        
                        result = np.full_like(chunk1, nodata_value, dtype=np.float32)
                        result[mask_both] = (chunk1[mask_both] - chunk2[mask_both]).astype(np.float32)
                        
                        dst.write(result, 1, window=((row_start, row_end), (0, width)))
    
    return {
        'merged_dem': output_merged_path,
        'tin_raster': tin_raster_path,
        'difference_raster': difference_path,
        'output_dir': output_dir,
        'temp_dir': temp_dir
    }

# ИЗВЛЕЧЕНИЕ ВЫСОТ ЗДАНИЙ 

def extract_building_heights(
    buildings_path: str,
    raster_path: str,
    output_csv: str,
    max_distance: float = 0.002,
    min_neighbors: int = 2,
    id_column: str = 'id',
    use_tqdm: bool = True
) -> pd.DataFrame:
    """
    Извлечение максимальных высот зданий из DEM растра
    
    Args:
        buildings_path: путь к GeoJSON со зданиями
        raster_path: путь к DEM растру
        output_csv: путь для сохранения CSV
        max_distance: максимальное расстояние для поиска соседей
        min_neighbors: минимальное количество соседей
        id_column: колонка с идентификатором
        use_tqdm: показывать прогресс
    
    Returns:
        DataFrame с атрибутами зданий
    """
    
    buildings = gpd.read_file(buildings_path)
    
    with rasterio.open(raster_path) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs
    
    if buildings.crs != raster_crs:
        buildings = buildings.to_crs(raster_crs)
    
    stats = zonal_stats(
        buildings,
        raster_path,
        stats=['max'],
        nodata=-9999
    )
    
    buildings['max_original'] = [s['max'] for s in stats]

    valid_count = buildings['max_original'].notna().sum()
    
    problem_mask = buildings['max_original'].isna() | (buildings['max_original'] < 0)
    good_mask = ~problem_mask
    
    problem_buildings = buildings[problem_mask].copy()
    good_buildings = buildings[good_mask].copy()

    if len(problem_buildings) > 0 and len(good_buildings) > 0:
        
        good_centers = []
        for idx, building in good_buildings.iterrows():
            centroid = building.geometry.centroid
            good_centers.append([centroid.x, centroid.y, building['max_original']])
        
        good_centers = np.array(good_centers)
        tree = cKDTree(good_centers[:, :2])
        
        filled_values = []
        filled_by_neighbors = 0
        filled_by_closest = 0
        
        iterator = problem_buildings.iterrows()
        if use_tqdm:
            iterator = tqdm(list(iterator), desc="   Заполнение")
        
        for idx, building in iterator:
            centroid = building.geometry.centroid
            indices = tree.query_ball_point([centroid.x, centroid.y], r=max_distance)
            
            if len(indices) >= min_neighbors:
                mean_value = np.mean(good_centers[indices, 2])
                filled_values.append(mean_value)
                filled_by_neighbors += 1
            elif len(indices) > 0:
                dist, nearest_idx = tree.query([[centroid.x, centroid.y]], k=1)
                filled_values.append(good_centers[nearest_idx[0], 2])
                filled_by_closest += 1
            else:
                filled_values.append(np.nan)
        
        problem_buildings['max_filled'] = filled_values
    
        remaining_mask = problem_buildings['max_filled'].isna()
        remaining_count = remaining_mask.sum()
        
        if remaining_count > 0:
            global_mean = good_buildings['max_original'].mean()
            problem_buildings.loc[remaining_mask, 'max_filled'] = global_mean

    good_buildings['max_filled'] = good_buildings['max_original']
    buildings_all = pd.concat([good_buildings, problem_buildings])
    buildings_all['max_value'] = buildings_all['max_filled']
    
    buildings_all['geometry_wkt'] = buildings_all.geometry.apply(lambda x: x.wkt)
    keep_columns = [col for col in buildings_all.columns if col not in ['geometry', 'max_original', 'max_filled']]
    attributes_df = buildings_all[keep_columns]
    
    attributes_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    return attributes_df

# ОБЪЕДИНЕНИЕ С ДРУГИМ CSV 

def merge_with_target_csv(
    source_csv: str,
    target_csv: str,
    output_csv: str,
    id_column: str = 'id'
) -> pd.DataFrame:
    """
    Объединение полученных высот с целевым CSV по ID
    
    Args:
        source_csv: CSV с вычисленными высотами
        target_csv: целевой CSV для добавления колонки
        output_csv: выходной CSV
        id_column: колонка с ID
    
    Returns:
        объединенный DataFrame
    """

    source = pd.read_csv(source_csv)
    target = pd.read_csv(target_csv)
    

    height_dict = dict(zip(source[id_column], source['max_value']))
    target['max_value'] = target[id_column].map(height_dict)
    
    filled = target['max_value'].notna().sum()
    
    target.to_csv(output_csv, index=False, encoding='utf-8')
    
    return target

# ВЫЧИТАНИЕ ДВУХ ГОТОВЫХ DEM

def subtract_two_dems(
    dem1_path: str,
    dem2_path: str,
    output_path: str,
    nodata_value: float = -9999.0,
    chunk_size: int = 1000
) -> str:
    """
    Вычитание двух готовых DEM (для Copernicus/FABDEM)
    """
    
    with rasterio.open(dem1_path) as src1, rasterio.open(dem2_path) as src2:

        overlap = [
            max(src1.bounds.left, src2.bounds.left),
            max(src1.bounds.bottom, src2.bounds.bottom),
            min(src1.bounds.right, src2.bounds.right),
            min(src1.bounds.top, src2.bounds.top)
        ]

        if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
            print(" Растры не перекрываются")
            return None

        def get_window(src, overlap):
            col_start = int((overlap[0] - src.transform.c) / src.transform.a)
            col_end = int((overlap[2] - src.transform.c) / src.transform.a) + 1
            row_start = int((src.transform.f - overlap[3]) / (-src.transform.e))
            row_end = int((src.transform.f - overlap[1]) / (-src.transform.e)) + 1
            return row_start, row_end, col_start, col_end
        
        row1_start, row1_end, col1_start, col1_end = get_window(src1, overlap)
        row2_start, row2_end, col2_start, col2_end = get_window(src2, overlap)
        
        height = min(row1_end - row1_start, row2_end - row2_start)
        width = min(col1_end - col1_start, col2_end - col2_start)
        
        out_transform = rasterio.Affine(
            src1.transform.a, 0, overlap[0],
            0, src1.transform.e, overlap[3]
        )
        
        profile = src1.profile.copy()
        profile.update({
            'height': height,
            'width': width,
            'transform': out_transform,
            'dtype': 'float32',
            'nodata': nodata_value,
            'compress': 'lzw', 
            'BIGTIFF': 'YES' 
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for row_start in tqdm(range(0, height, chunk_size), desc="   Вычитание"):
                row_end = min(row_start + chunk_size, height)
                
                window1 = ((row1_start + row_start, row1_start + row_end),
                          (col1_start, col1_start + width))
                window2 = ((row2_start + row_start, row2_start + row_end),
                          (col2_start, col2_start + width))
                
                chunk1 = src1.read(1, window=window1).astype('float32')
                chunk2 = src2.read(1, window=window2).astype('float32')
                
                valid1 = ~np.isnan(chunk1) & (chunk1 != src1.nodata if src1.nodata else True)
                valid2 = ~np.isnan(chunk2) & (chunk2 != src2.nodata if src2.nodata else True)
                
                mask_both = valid1 & valid2
                
                result = np.full_like(chunk1, nodata_value, dtype='float32')
                result[mask_both] = chunk1[mask_both] - chunk2[mask_both]
                
                dst.write(result, 1, window=((row_start, row_end), (0, width)))
    
    return output_path


def process_arcticdem_final(
    dem_tiles_pattern: str,
    roads_path: str,
    output_dir: str,
    temp_dir: str,
    buffer_meters: float = 100,
    nodata_value: float = -9999.0,
    chunk_size_interp: int = 500,
    chunk_size_io: int = 1000,
    use_tqdm: bool = True
) -> dict:
    """
    Финальная версия обработки ArcticDEM - запись по блокам
    """
    import shutil
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    output_merged_path = os.path.join(output_dir, "dem_merged.tif")
    
    tile_paths = glob.glob(dem_tiles_pattern)
    if not tile_paths:
        raise FileNotFoundError(f"Тайлы не найдены: {dem_tiles_pattern}")
    
    print(f"   Найдено тайлов: {len(tile_paths)}")

    # Загружаем дороги и перепроецируем
    roads = gpd.read_file(roads_path)
    
    with rasterio.open(tile_paths[0]) as src:
        raster_crs = src.crs
        raster_transform = src.transform
        raster_width = src.width
        raster_height = src.height
    
    if roads.crs != raster_crs:
        print(f"   Перепроецирование дорог из {roads.crs} в {raster_crs}")
        roads = roads.to_crs(raster_crs)
    
    # Получаем общий bounding box дорог с буфером
    bounds = roads.total_bounds
    minx = bounds[0] - buffer_meters
    miny = bounds[1] - buffer_meters
    maxx = bounds[2] + buffer_meters
    maxy = bounds[3] + buffer_meters
    
    # Вычисляем общее окно
    col_min = max(0, int((minx - raster_transform[2]) / raster_transform[0]))
    col_max = min(raster_width, int((maxx - raster_transform[2]) / raster_transform[0]) + 1)
    row_min = max(0, int((maxy - raster_transform[5]) / raster_transform[4]))
    row_max = min(raster_height, int((miny - raster_transform[5]) / raster_transform[4]) + 1)
    
    out_width = col_max - col_min
    out_height = row_max - row_min
    
    print(f"   Область обработки: строки {row_min}-{row_max}, столбцы {col_min}-{col_max}")
    print(f"   Размер выходного растра: {out_width} x {out_height}")
    
    # Создаем выходной растр (без предварительного заполнения массива!)
    out_transform = rasterio.transform.from_origin(minx, maxy, raster_transform[0], -raster_transform[4])
    
    profile = {
        'driver': 'GTiff',
        'height': out_height,
        'width': out_width,
        'count': 1,
        'dtype': 'float32',
        'crs': raster_crs,
        'transform': out_transform,
        'compress': 'lzw',
        'nodata': nodata_value,
        'BIGTIFF': 'YES'
    }
    
    # Создаем пустой растр (без записи данных)
    with rasterio.open(output_merged_path, 'w', **profile) as dst:
        pass
    
    # Обрабатываем каждый тайл и записываем блоками
    block_size = 500
    
    iterator = enumerate(tile_paths)
    if use_tqdm:
        iterator = tqdm(list(iterator), desc="   Обработка тайлов")
    
    for i, tile_path in iterator:
        with rasterio.open(tile_path) as src:
            # Вычисляем окно в этом тайле
            tile_col_min = max(0, int((minx - src.transform[2]) / src.transform[0]))
            tile_col_max = min(src.width, int((maxx - src.transform[2]) / src.transform[0]) + 1)
            tile_row_min = max(0, int((maxy - src.transform[5]) / src.transform[4]))
            tile_row_max = min(src.height, int((miny - src.transform[5]) / src.transform[4]) + 1)
            
            if tile_col_min >= tile_col_max or tile_row_min >= tile_row_max:
                continue
            
            # Читаем и записываем блоками
            for row_start in range(tile_row_min, tile_row_max, block_size):
                row_end = min(row_start + block_size, tile_row_max)
                for col_start in range(tile_col_min, tile_col_max, block_size):
                    col_end = min(col_start + block_size, tile_col_max)
                    
                    window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    try:
                        data = src.read(1, window=window)
                    except Exception as e:
                        print(f"   Ошибка чтения блока: {e}")
                        continue
                    
                    # Вычисляем положение в выходном растре
                    out_col_start = col_start - col_min
                    out_row_start = row_start - row_min
                    
                    # Записываем блок
                    with rasterio.open(output_merged_path, 'r+') as dst:
                        dst_window = rasterio.windows.Window(
                            out_col_start, out_row_start,
                            data.shape[1], data.shape[0]
                        )
                        dst.write(data.astype('float32'), 1, window=dst_window)
    
    print(f"\n   Результат: {output_merged_path}")
    
    # ... остальной код построения ЦМР, TIN и вычитания ...
    
    # 7. ПОСТРОЕНИЕ ЦМР (остается без изменений)
    print("\n ПОСТРОЕНИЕ ЦМР")
    
    with rasterio.open(output_merged_path) as src:
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata
        bounds = src.bounds
        original_width = src.width
        original_height = src.height
    
    x_coords = transform.c + np.arange(original_width) * transform.a
    y_coords = transform.f + np.arange(original_height) * transform.e

    roads = gpd.read_file(roads_path)
    if raster_crs is not None and roads.crs != raster_crs:
        roads = roads.to_crs(raster_crs)
    
    all_lines = []
    for geom in roads.geometry:
        if geom.geom_type == 'LineString':
            all_lines.append(geom)
        elif geom.geom_type == 'MultiLineString':
            all_lines.extend(list(geom.geoms))
        

    lines = all_lines
    all_points_set = set()

    iterator = range(len(lines))
    if use_tqdm:
        iterator = tqdm(iterator, desc="   Пересечения")
    
    for i in iterator:
        for j in range(i+1, len(lines)):
            if lines[i].intersects(lines[j]):
                intersection = lines[i].intersection(lines[j])
                if intersection.geom_type == 'Point':
                    all_points_set.add((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        all_points_set.add((point.x, point.y))
    
    iterator = lines
    if use_tqdm:
        iterator = tqdm(iterator, desc="   Конечные точки")
    
    for line in iterator:
        if line.geom_type == 'LineString':
            coords = list(line.coords)
            if len(coords) >= 2:
                all_points_set.add(coords[0])
                all_points_set.add(coords[-1])
    
    all_points = list(all_points_set)
    
    valid_points = []
    for point in all_points:
        if (bounds.left <= point[0] <= bounds.right and 
            bounds.bottom <= point[1] <= bounds.top):
            valid_points.append(point)

    points_with_height = []
    
    with rasterio.open(output_merged_path) as src:
        iterator = valid_points
        if use_tqdm:
            iterator = tqdm(iterator, desc="   Извлечение высот")
        
        for point in iterator:
            col = int((point[0] - transform.c) / transform.a)
            row = int((point[1] - transform.f) / transform.e)
            
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1, window=((row, row+1), (col, col+1)))[0,0]
                
                if nodata is None or value != nodata:
                    points_with_height.append([point[0], point[1], value])
    
    points = np.array(points_with_height)
    
    if len(points) < 3:
        print("Недостаточно точек для TIN (менее 3)")
        tin_raster_path = None
        difference_path = None
    else:
        tri = Delaunay(points[:, :2])
        
        interpolator = LinearNDInterpolator(tri, points[:, 2])
        tin_raster_path = os.path.join(temp_dir, "tin_raster.tif")
        
        profile_tin = {
            'driver': 'GTiff',
            'height': original_height,
            'width': original_width,
            'count': 1,
            'dtype': np.float32,
            'crs': raster_crs,
            'transform': transform,
            'nodata': nodata_value,
            'compress': 'lzw',
            'BIGTIFF': 'YES' 
        }
        
        x_coords = transform.c + np.arange(original_width) * transform.a
        y_coords = transform.f + np.arange(original_height) * transform.e
        
        with rasterio.open(tin_raster_path, 'w', **profile_tin) as dst:
            block_size = chunk_size_interp
            rows_range = range(0, original_height, block_size)
            if use_tqdm:
                rows_range = tqdm(rows_range, desc="   Интерполяция")
            
            for row_start in rows_range:
                row_end = min(row_start + block_size, original_height)
                
                for col_start in range(0, original_width, block_size):
                    col_end = min(col_start + block_size, original_width)
                    
                    block_x, block_y = np.meshgrid(
                        x_coords[col_start:col_end],
                        y_coords[row_start:row_end]
                    )
                    
                    block_values = interpolator(block_x, block_y)
                    block_values = np.where(np.isnan(block_values), nodata_value, block_values)
                    
                    dst.write(
                        block_values.astype(np.float32),
                        1,
                        window=((row_start, row_end), (col_start, col_end))
                    )
        
        difference_path = os.path.join(output_dir, "difference.tif")
        
        with rasterio.open(output_merged_path) as src1, rasterio.open(tin_raster_path) as src2:
            
            overlap = [
                max(src1.bounds.left, src2.bounds.left),
                max(src1.bounds.bottom, src2.bounds.bottom),
                min(src1.bounds.right, src2.bounds.right),
                min(src1.bounds.top, src2.bounds.top)
            ]

            if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
                print("Растры не перекрываются")
                difference_path = None
            else:
                col_min1 = int((overlap[0] - src1.transform.c) / src1.transform.a)
                col_max1 = int((overlap[2] - src1.transform.c) / src1.transform.a) + 1
                row_min1 = int((src1.transform.f - overlap[3]) / (-src1.transform.e))
                row_max1 = int((src1.transform.f - overlap[1]) / (-src1.transform.e)) + 1
                
                col_min2 = int((overlap[0] - src2.transform.c) / src2.transform.a)
                col_max2 = int((overlap[2] - src2.transform.c) / src2.transform.a) + 1
                row_min2 = int((src2.transform.f - overlap[3]) / (-src2.transform.e))
                row_max2 = int((src2.transform.f - overlap[1]) / (-src2.transform.e)) + 1
                
                width = min(col_max1 - col_min1, col_max2 - col_min2)
                height = min(row_max1 - row_min1, row_max2 - row_min2)
                
                out_transform = rasterio.Affine(
                    src1.transform.a, 0, overlap[0],
                    0, src1.transform.e, overlap[3]
                )
                
                profile_diff = src1.profile.copy()
                profile_diff.update({
                    'height': height,
                    'width': width,
                    'transform': out_transform,
                    'dtype': 'float32',
                    'nodata': nodata_value,
                    'compress': 'lzw',
                    'BIGTIFF': 'YES' 
                })
                
                with rasterio.open(difference_path, 'w', **profile_diff) as dst:
                    chunk_size = chunk_size_io
                    rows_range = range(0, height, chunk_size)
                    if use_tqdm:
                        rows_range = tqdm(rows_range, desc="   Вычитание")
                    
                    for row_start in rows_range:
                        row_end = min(row_start + chunk_size, height)
                        
                        window1 = ((row_min1 + row_start, row_min1 + row_end), 
                                  (col_min1, col_min1 + width))
                        window2 = ((row_min2 + row_start, row_min2 + row_end), 
                                  (col_min2, col_min2 + width))
                        
                        chunk1 = src1.read(1, window=window1)
                        chunk2 = src2.read(1, window=window2)
                        
                        if src1.nodata is not None:
                            valid1 = (chunk1 != src1.nodata) & (~np.isnan(chunk1))
                        else:
                            valid1 = ~np.isnan(chunk1)
                            
                        if src2.nodata is not None:
                            valid2 = (chunk2 != src2.nodata) & (~np.isnan(chunk2))
                        else:
                            valid2 = ~np.isnan(chunk2)
                        
                        mask_both = valid1 & valid2
                        
                        result = np.full_like(chunk1, nodata_value, dtype=np.float32)
                        result[mask_both] = (chunk1[mask_both] - chunk2[mask_both]).astype(np.float32)
                        
                        dst.write(result, 1, window=((row_start, row_end), (0, width)))
    
    return {
        'merged_dem': output_merged_path,
        'tin_raster': tin_raster_path,
        'difference_raster': difference_path,
        'output_dir': output_dir,
        'temp_dir': temp_dir
    }
    

# ПОЛНЫЙ ПАЙПЛАЙН

def dem_full_pipeline(
    # Обязательные параметры
    roads_path: str,  # полный путь к дорогам
    buildings_path: str,  # полный путь к зданиям
    
    # Для ArcticDEM
    dem_tiles_pattern: Optional[str] = None,  # паттерн для тайлов ArcticDEM
    
    # Для Copernicus/FABDEM
    fabdem_tiles_pattern: Optional[str] = None,  # паттерн для тайлов FABDEM 
    copernicus_dem: Optional[str] = None,        # путь к Copenicus DEM 
    
    # Параметры обработки
    dem_type: str = 'arcticdem',  # 'arcticdem','fabdem'
    output_dir: Optional[str] = None,  # куда сохранять результаты
    output_csv: Optional[str] = None,  # куда сохранить CSV с высотами
    use_difference: bool = True,
    merge_with_target: bool = False,
    target_csv: Optional[str] = None,
    
    # Дополнительные параметры
    buffer_degrees: float = 100,
    nodata_value: float = -9999.0,
    chunk_size_interp: int = 500,
    chunk_size_io: int = 1000,
    use_tqdm: bool = True
) -> dict:
    """
    ПОЛНЫЙ ПАЙПЛАЙН: обработка DEM + извлечение высот зданий
    
    Args:
        roads_path: ПОЛНЫЙ путь к GeoJSON с дорогами
        buildings_path: ПОЛНЫЙ путь к GeoJSON со зданиями
        dem_tiles_pattern: паттерн для тайлов ArcticDEM (например "D:/data/*_dem.tif")
        fabdem_tiles_pattern: путь к файлам FABDEM 
        copernicus_dem: путь к файлу Copernicus DEM 
        dem_type: 'arcticdem' или 'fabdem'
        output_dir: директория для сохранения результатов
        output_csv: путь для сохранения CSV с высотами
        use_difference: использовать разницу или исходный DEM
        merge_with_target: объединять с целевым CSV
        target_csv: путь к целевому CSV
        buffer_degrees: буфер вокруг дорог
        nodata_value: значение NoData
        chunk_size_interp: размер блока для интерполяции
        chunk_size_io: размер блока для ввода-вывода
        use_tqdm: показывать прогресс
    
    Returns:
        словарь с результатами
    """

    if output_dir is None:
        base_dir = os.path.dirname(buildings_path)
        city_name = os.path.basename(base_dir)
        output_dir = os.path.join(base_dir, f"{city_name}_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if output_csv is None:
        output_csv = os.path.join(output_dir, "building_heights.csv")
    
    # РАСТР ДЛЯ ИЗВЛЕЧЕНИЯ ВЫСОТ 
    raster_for_buildings = None
    dem_results = None
    
    # Обработка ARCTICDEM 
    if dem_type == 'arcticdem':
        if dem_tiles_pattern is None:
            raise ValueError("Для ArcticDEM необходимо указать dem_tiles_pattern")
        
        print(f"\n Обработка ArcticDEM")

        temp_arctic_dir = os.path.join(output_dir, 'temp_arctic')
        os.makedirs(temp_arctic_dir, exist_ok=True)
        
        dem_results = process_arcticdem(
            dem_tiles_pattern=dem_tiles_pattern,
            roads_path=roads_path,
            output_dir=output_dir,
            temp_dir=temp_arctic_dir,
            buffer_degrees=buffer_degrees,
            nodata_value=nodata_value,
            chunk_size_interp=chunk_size_interp,
            chunk_size_io=chunk_size_io,
            use_tqdm=use_tqdm
        )
        if dem_results is None:
            return {'status': 'error', 'message': 'DEM processing failed'}
        
        if use_difference:
            raster_for_buildings = dem_results['difference_raster']
            if raster_for_buildings is None or not os.path.exists(raster_for_buildings):
                raise FileNotFoundError(f" Растр разницы не создан")
        else:
            raster_for_buildings = dem_results['merged_dem']
    
    # Обработка COPERNICUS/FABDEM 
    elif dem_type == 'fabdem':
        if copernicus_dem is None:
            raise ValueError("Для FABDEM необходимо указать copernicus_dem")
        if fabdem_tiles_pattern is None:
            raise ValueError("Для FABDEM необходимо указать fabdem_tiles_pattern")

        import glob
        fabdem_tiles = glob.glob(fabdem_tiles_pattern)
        if not fabdem_tiles:
            raise FileNotFoundError(f"Тайлы FABDEM не найдены: {fabdem_tiles_pattern}")
        
        if use_difference:
            from rasterio.merge import merge
            
            src_files = [rasterio.open(tile) for tile in fabdem_tiles]
            fabdem_mosaic, fabdem_transform = merge(src_files)
            
            for src in src_files:
                src.close()

            fabdem_merged_path = os.path.join(output_dir, "fabdem_merged.tif")
            profile = src_files[0].profile.copy() if src_files else {}
            profile.update({
                'height': fabdem_mosaic.shape[1],
                'width': fabdem_mosaic.shape[2],
                'transform': fabdem_transform,
                'compress': 'lzw', 
                'BIGTIFF': 'YES' 
            })
            
            with rasterio.open(fabdem_merged_path, 'w', **profile) as dst:
                dst.write(fabdem_mosaic, 1)

            diff_dir = os.path.join(output_dir, 'difference')
            os.makedirs(diff_dir, exist_ok=True)
            
            base1 = os.path.basename(copernicus_dem).replace('.tif', '')
            difference_path = os.path.join(diff_dir, f"{base1}_minus_fabdem.tif")
            
            print(f"   Вычитание DEM: {base1} - FABDEM")
            raster_for_buildings = subtract_two_dems(
                dem1_path=copernicus_dem,
                dem2_path=fabdem_merged_path,
                output_path=difference_path,
                nodata_value=nodata_value,
                chunk_size=chunk_size_io
            )
            
            dem_results = {
                'fabdem_tiles': fabdem_tiles,
                'fabdem_merged': fabdem_merged_path,
                'copernicus_dem': copernicus_dem,
                'difference_raster': difference_path
            }
        else:
            raster_for_buildings = copernicus_dem
            dem_results = {'merged_dem': copernicus_dem}
    
    # ИЗВЛЕЧЕНИЕ ВЫСОТ ЗДАНИЙ
    print(f"\n Извлечение высот зданий")
    
    if raster_for_buildings and os.path.exists(raster_for_buildings):
        buildings_df = extract_building_heights(
            buildings_path=buildings_path,
            raster_path=raster_for_buildings,
            output_csv=output_csv,
            use_tqdm=use_tqdm
        )
    else:
        raise FileNotFoundError(f" Растр для извлечения не найден: {raster_for_buildings}")
    
    # ОБЪЕДИНЕНИЕ С ЦЕЛЕВЫМ CSV
    final_csv = None
    if merge_with_target and target_csv and os.path.exists(target_csv):
        final_csv = os.path.join(output_dir, "buildings_ready.csv")
        merge_with_target_csv(
            source_csv=output_csv,
            target_csv=target_csv,
            output_csv=final_csv
        )

    return {
        'status': 'success',
        'dem_type': dem_type,
        'dem_results': dem_results,
        'raster_used': raster_for_buildings,
        'buildings_csv': output_csv,
        'final_csv': final_csv
    }


