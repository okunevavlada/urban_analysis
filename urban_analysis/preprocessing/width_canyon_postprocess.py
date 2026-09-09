"""
Модуль для обработки растра ширины каньона: обрезка, заполнение NoData, фильтрация
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from typing import Optional, List, Union, Dict
os.environ['USE_PYGEOS'] = '0'
import whitebox_workflows as wbw
wbe = wbw.WbEnvironment()


def extract_band(
    input_raster: str,
    output_raster: str,
    band: int = 1
) -> str:
    """
    Извлечение одного канала из многоканального растра
    
    Args:
        input_raster: путь к входному многоканальному растру
        output_raster: путь для сохранения одноканального растра
        band: номер канала для извлечения
    
    Returns:
        str: путь к сохраненному растру
    """
    with rasterio.open(input_raster) as src:
        array = src.read(band)
        profile = src.profile.copy()
        
        profile.update({
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(array, 1)
    
    return output_raster


def fill_nodata_raster(
    input_raster: str,
    output_raster: str,
    filter_size: int = 50
) -> str:
    """
    Заполнение ячеек NoData в растре методом интерполяции
    
    Args:
        input_raster: путь к входному растру с NoData значениями
        output_raster: путь для сохранения заполненного растра
        filter_size: размер фильтра в пикселях для интерполяции пропусков
    
    Returns:
        str: путь к сохраненному заполненному растру
    """
    input_data = wbe.read_raster(input_raster)
    nodata_filled = wbe.fill_missing_data(input_data, filter_size=filter_size)
    wbe.write_raster(nodata_filled, output_raster)
    return output_raster


def mean_filter_raster(
    input_raster: str,
    output_raster: str,
    filter_size: int = 100
) -> str:
    """
    Фильтрация растра 
    
    Args:
        input_raster: путь к входному растру
        output_raster: путь для сохранения отфильтрованного растра
        filter_size: размер фильтра в пикселях

    Returns:
        str: путь к сохраненному отфильтрованному растру
    """
    input_data = wbe.read_raster(input_raster)
    filtered = wbe.mean_filter(input_data, filter_size)
    wbe.write_raster(filtered, output_raster)
    return output_raster


def clip_raster_by_polygons(
    input_raster: str,
    output_raster: str,
    polygons: Union[str, gpd.GeoDataFrame],
    invert: bool = True,
    nodata_value: int = 0,
    dtype: str = 'int16',
    compress: str = 'lzw'
) -> str:
    """
    Обрезка растра по полигональным данным
    
    Args:
        input_raster: путь к входному растровому файлу
        output_raster: путь для сохранения обрезанного растра
        polygons: путь к GeoJSON файлу или GeoDataFrame с полигонами для обрезки
        invert: если True - оставить область внутри полигонов, если False - вне полигонов
        nodata_value: значение NoData для выходного растра
        dtype: тип данных выходного растра
        compress: метод сжатия
    
    Returns:
        str: путь к сохраненному обрезанному растру
    """
    if isinstance(polygons, str):
        gdf = gpd.read_file(polygons)
    else:
        gdf = polygons.copy()
    
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        
        array = src.read(1).astype(dtype)
        profile = src.profile.copy()
        
        out_image, out_transform = mask(
            src, 
            gdf.geometry, 
            invert=invert, 
            nodata=nodata_value, 
            filled=False
        )
        
        profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform,
            'nodata': nodata_value,
            'count': 1,
            'compress': compress,
            'dtype': dtype
        })
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(out_image[0], indexes=1)
    
    return output_raster


def extract_zonal_stats(
    polygons: Union[str, gpd.GeoDataFrame],
    raster_path: str,
    output_csv: Optional[str] = None,
    stats: List[str] = ['mean'],
    id_column: str = 'id',
    fallback_to_centroid: bool = True
) -> gpd.GeoDataFrame:
    """
    Извлечение зональной статистики с прогрессом
    
    Args:
        polygons: путь к GeoJSON файлу или GeoDataFrame с полигонами зданий
        raster_path: путь к растру
        output_csv: путь для сохранения CSV файла с результатами
        stats: список статистик для извлечения
        id_column: название колонки с идентификатором здания
        fallback_to_centroid: заменять NaN на значение в центроиде
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame с добавленными колонками статистик
    """
    from rasterstats import zonal_stats
    import rasterio
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    if isinstance(polygons, str):
        gdf = gpd.read_file(polygons)
    else:
        gdf = polygons.copy()
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        chunk_size = 100
        total = len(gdf)
        all_stats = []

        with tqdm(total=total, desc="         Обработка зданий", unit="зд.") as pbar:
            for i in range(0, total, chunk_size):
                chunk = gdf.iloc[i:i+chunk_size]

                chunk_stats = zonal_stats(
                    chunk,
                    raster_path,
                    stats=stats,
                    nodata=0
                )
                all_stats.extend(chunk_stats)

                pbar.update(len(chunk))

        for stat in stats:
            gdf['width_med'] = [s[stats[0]] if s is not None and s.get(stats[0]) is not None else np.nan for s in all_stats]

        if fallback_to_centroid:
            transform = src.transform
            nodata = src.nodata if src.nodata is not None else 0
            
            nan_indices = gdf.index[gdf['width_med'].isna()].tolist()
            
            if nan_indices:
                
                for idx in tqdm(nan_indices, desc="         Замена NaN", unit="зд."):
                    row = gdf.loc[idx]
                    centroid = row.geometry.centroid
                    x, y = centroid.x, centroid.y
                    
                    col = int((x - transform.c) / transform.a)
                    row_idx = int((y - transform.f) / transform.e)
                    
                    if 0 <= row_idx < src.height and 0 <= col < src.width:
                        value = src.read(1, window=((row_idx, row_idx+1), (col, col+1)))[0,0]
                        if value != nodata:
                            gdf.at[idx, 'width_med'] = value
    
    if output_csv:
        attrs_df = gdf.drop(columns=['geometry'])
        attrs_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    return gdf


def process_width_raster_pipeline(
    width_raster: str,
    polygons: Union[str, gpd.GeoDataFrame],
    output_dir: str,
    band: int = 2,
    clip_raster_name: str = "width_clip.tif",
    filled_raster_name: str = "width_filled.tif",
    filtered_raster_name: str = "width_filled_filter.tif",
    output_csv_name: str = "buildings_width.csv",
    invert_mask: bool = True,
    nodata_value: int = 0,
    fill_filter_size: int = 50,
    mean_filter_size: int = 100, 
    zonal_stats_list: Union[str, List[str]] = 'mean',
    id_column: str = 'id',
    fallback_to_centroid: bool = True
) -> Dict[str, str]:
    """
    ПОЛНЫЙ ПАЙПЛАЙН ОБРАБОТКИ РАСТРА ШИРИНЫ
    
    Args:
        width_raster: путь к исходному растру ширины (может быть многоканальным)
        polygons: путь к GeoJSON файлу или GeoDataFrame с полигонами зданий
        output_dir: директория для сохранения всех результатов
        band: номер канала для извлечения (по умолчанию 2)
        clip_raster_name: имя файла для обрезанного растра
        filled_raster_name: имя файла для растра с заполненными NoData
        filtered_raster_name: имя файла для отфильтрованного растра
        output_csv_name: имя файла для CSV с результатами
        invert_mask: если True - оставить область внутри полигонов
        nodata_value: значение NoData для растров
        fill_filter_size: размер фильтра для заполнения NoData (в пикселях)
        mean_filter_size: размер фильтра для сглаживания (в пикселях)
        zonal_stats_list: список статистик или одна статистика ('mean', 'max', 'min', 'sum', 'std')
        id_column: название колонки с идентификатором здания
        fallback_to_centroid: заменять NaN на значение в центроиде
    
    Returns:
        Dict[str, str]: словарь с путями к созданным файлам
    """
    
    if isinstance(zonal_stats_list, str):
        zonal_stats_list = [zonal_stats_list]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Извлечение нужного канала
    temp_band = os.path.join(output_dir, "_temp_band.tif")
    extract_band(width_raster, temp_band, band)
    
    clip_path = os.path.join(output_dir, clip_raster_name)
    filled_path = os.path.join(output_dir, filled_raster_name)
    filtered_path = os.path.join(output_dir, filtered_raster_name)
    csv_path = os.path.join(output_dir, output_csv_name)

    print("\n1. Обрезка растра")
    clip_raster = clip_raster_by_polygons(
        input_raster=temp_band,
        output_raster=clip_path,
        polygons=polygons,
        invert=invert_mask,
        nodata_value=nodata_value
    )
    
    print("\n2. Заполнение пропусков")
    filled_raster = fill_nodata_raster(
        input_raster=clip_path,
        output_raster=filled_path,
        filter_size=fill_filter_size
    )
    
    print("\n3. Сглаживание")
    filtered_raster = mean_filter_raster(
        input_raster=filled_path,
        output_raster=filtered_path,
        filter_size=mean_filter_size
    )
    
    print("\n4. Извлечение статистики")
    gdf_result = extract_zonal_stats(
        polygons=polygons,
        raster_path=filtered_path,
        output_csv=csv_path,
        stats=zonal_stats_list,
        id_column=id_column,
        fallback_to_centroid=fallback_to_centroid
    )
    
    # Удаление временного файла
    if os.path.exists(temp_band):
        os.remove(temp_band)
    
    return {
        'clip_raster': clip_path,
        'filled_raster': filled_path,
        'filtered_raster': filtered_path,
        'output_csv': csv_path
    }