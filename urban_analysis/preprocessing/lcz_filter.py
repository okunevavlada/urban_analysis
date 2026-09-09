"""
Модуль для обработки растра Локальных климатических зон (ЛКЗ) 
"""

import os
import tempfile
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import rowcol
import geopandas as gpd
from scipy import stats
from tqdm import tqdm
from typing import Optional, List, Dict
import whitebox_workflows as wbw
from shapely.geometry import box

os.environ['USE_PYGEOS'] = '0'
wbe = wbw.WbEnvironment()


def process_lcz_raster(
    lcz_raster: str,
    buildings_path: Optional[str] = None,  
    output_raster: Optional[str] = None,
    remove_values: List[int] = None,
    fill_filter_size: int = 3,
    band_number: int = 1  
) -> str:
    """
    Обработка растра ЛКЗ:
    1. Обрезка по ограничивающему прямоугольнику зданий (если передан buildings_path)
    2. Удаление пикселей с заданными значениями
    3. Заполнение пропусков из соседних ячеек

    Args:
        lcz_raster: путь к входному растру ЛКЗ
        buildings_path: путь к GeoJSON со зданиями (опционально, для обрезки)
        output_raster: путь для сохранения
        remove_values: список значений для удаления
        fill_filter_size: размер фильтра для заполнения пропусков
        band_number: номер канала (с 1), который нужно обработать

    Returns:
        путь к обработанному растру
    """
    if remove_values is None:
        remove_values = list(range(11, 18))

    if output_raster is None:
        base, ext = os.path.splitext(lcz_raster)

    raster_to_process = lcz_raster
    temp_cropped_raster = None
    
    if buildings_path is not None and os.path.exists(buildings_path):

        gdf = gpd.read_file(buildings_path)
        buildings_bounds = gdf.total_bounds
        bbox_polygon = box(buildings_bounds[0], buildings_bounds[1], buildings_bounds[2], buildings_bounds[3])
        buildings_bbox = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=gdf.crs)
        
        with rasterio.open(lcz_raster) as src:
            if gdf.crs != src.crs:
                bbox_proj = buildings_bbox.to_crs(src.crs)
                bbox_geom = bbox_proj.geometry.values[0]
            else:
                bbox_geom = buildings_bbox.geometry.values[0]

            out_image, out_transform = mask(src, [bbox_geom], crop=True)

            temp_cropped_raster = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_cropped_raster.close()
            
            profile = src.profile.copy()
            profile.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform,
                'count': out_image.shape[0]
            })
            
            with rasterio.open(temp_cropped_raster.name, 'w', **profile) as dst:
                dst.write(out_image)

            raster_to_process = temp_cropped_raster.name

    with rasterio.open(raster_to_process) as src:
        data = src.read(band_number).astype(np.int16)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else 0

    mask_remove = np.isin(data, remove_values)
    data_cleaned = data.copy()
    data_cleaned[mask_remove] = nodata

    temp_path = output_raster.replace('.tif', '_temp.tif')

    profile.update({
        'dtype': 'int16',
        'nodata': nodata,
        'count': 1 
    })

    with rasterio.open(temp_path, 'w', **profile) as dst:
        dst.write(data_cleaned, 1)

    input_data = wbe.read_raster(temp_path)
    filled_data = wbe.fill_missing_data(input_data, filter_size=fill_filter_size)
    wbe.write_raster(filled_data, output_raster)

    os.remove(temp_path)

    if temp_cropped_raster and os.path.exists(temp_cropped_raster.name):
        os.unlink(temp_cropped_raster.name)

    return output_raster

def extract_majority_values(
    buildings_path: str,
    raster_path: str,
    output_csv: str,
    buffer_meters: float = 100,
    crs_code: Optional[int] = None, 
    id_column: str = 'id',
    nodata_value: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Извлечение модальных значений из растра для зданий с буфером
    
    Args:
        buildings_path: путь к GeoJSON со зданиями
        raster_path: путь к растру (обработанному LCZ)
        output_csv: путь для сохранения CSV
        buffer_meters: размер буфера в метрах
        crs_code: целевой CRS код (например, 32645) - используется для перепроецирования
        id_column: колонка с идентификатором
        nodata_value: значение NoData (если None, берется из растра)
    
    Returns:
        GeoDataFrame с добавленными значениями
    """
    import tempfile
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    gdf = gpd.read_file(buildings_path)

    with rasterio.open(raster_path) as src:
        if nodata_value is None:
            nodata_value = src.nodata if src.nodata is not None else 0
        raster_original_crs = src.crs

    if crs_code is not None:
        target_crs = f"EPSG:{crs_code}"
    
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    temp_raster = None
    
    with rasterio.open(raster_path) as src:
        if src.crs != target_crs:
            
            temp_raster = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_raster.close()
            
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rasterio.open(temp_raster.name, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest)
            
            raster_to_use = temp_raster.name
        else:
            raster_to_use = raster_path

    buffer_units = buffer_meters
    
    result_column = f'majority_{int(buffer_meters)}m'
    gdf[result_column] = None
    
    with rasterio.open(raster_to_use) as src:
        iterator = gdf.iterrows()
        iterator = tqdm(iterator, total=len(gdf), desc="   Обработка зданий")
        
        for idx, row in iterator:
            geom = row.geometry
            buffer_geom = geom.buffer(buffer_units)
            
            out_image, _ = mask(src, [buffer_geom], crop=True)
            band = out_image[0]
            valid_pixels = band[band != nodata_value]
            
            if valid_pixels.size > 0:
                mode_val = stats.mode(valid_pixels, axis=None, keepdims=False).mode
                gdf.at[idx, result_column] = str(int(mode_val))
            else:
                gdf.at[idx, result_column] = None

    attrs_df = gdf.drop(columns=['geometry'])
    attrs_df.to_csv(output_csv, index=False, encoding='utf-8')

    if temp_raster and os.path.exists(temp_raster.name):
        os.unlink(temp_raster.name)
    
    return gdf

def lcz_prepair(
    lcz_raster: str,
    buildings_path: str,
    output_dir: str,
    remove_values: List[int] = None,
    fill_filter_size: int = 3,
    buffer_meters: float = 100,
    processed_raster_name: str = "lcz_processed.tif",
    output_csv_name: str = "buildings_lcz.csv",
    band_number: int = 1,
    crs_code: Optional[int] = None
) -> Dict[str, str]:
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_raster = os.path.join(output_dir, processed_raster_name)
    output_csv = os.path.join(output_dir, output_csv_name)

    processed = process_lcz_raster(
        lcz_raster=lcz_raster,
        buildings_path=buildings_path,
        output_raster=processed_raster,
        remove_values=remove_values,
        fill_filter_size=fill_filter_size,
        band_number=band_number
    )

    gdf_result = extract_majority_values(
        buildings_path=buildings_path,
        raster_path=processed,
        output_csv=output_csv,
        buffer_meters=buffer_meters,
        crs_code=crs_code,  
        id_column='id',
        nodata_value=None
    )
    
    return {
        'processed_raster': processed,
        'output_csv': output_csv,
        'gdf': gdf_result
    }