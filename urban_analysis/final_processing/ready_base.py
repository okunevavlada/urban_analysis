"""
Функция создания готовой базы данных (упрощенная версия)
"""

def create_multiband_raster(
    output_path: str,
    template_raster: str,
    band_files: list,
    band_names: list,
    band_indices: list = None,
    nodata_value: int = -1,
    dtype: str = 'int16',
    compress: str = 'lzw',
    target_crs: str = None,  
    discrete_bands: list = None  
) -> str:
    """
    Создание многоканального растра 
    
    Args:
        output_path: путь для сохранения
        template_raster: путь к растру-шаблону
        band_files: список путей к файлам каналов
        band_names: список названий каналов
        band_indices: список индексов каналов 
        nodata_value: значение NoData
        dtype: тип данных
        compress: метод сжатия
        target_crs: целевая проекция (EPSG или WKT)
        discrete_bands: список номеров дискретных каналов 
            
    
    Returns:
        str: путь к созданному растру
    """
    import os
    import rasterio
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.enums import Resampling
    from rasterio.crs import CRS


    if band_indices is None:
        band_indices = [1] * len(band_files)

    if discrete_bands is None:
        discrete_bands_idx = []
    else:
        discrete_bands_idx = [i - 1 for i in discrete_bands]  

    with rasterio.open(template_raster) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        src_bounds = src.bounds
    
    if dst_crs is None:
        dst_crs = src_crs
        need_transform = False
    else:
        need_transform = dst_crs != src_crs

    arrays = []
    for i, (band_file, band_idx) in enumerate(zip(band_files, band_indices)):
        
        with rasterio.open(band_file) as src:
            array = src.read(band_idx).astype(dtype)
            if src.nodata is not None:
                array[array == src.nodata] = nodata_value
            arrays.append({
                'data': array,
                'crs': src.crs,
                'transform': src.transform
            })
    
        if need_transform:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height,
                *src_bounds
            )
        
        # Перепроецируем каждый канал
        transformed = []
        for i, band in enumerate(arrays):
            if i in discrete_bands_idx:
                resampling = Resampling.nearest
            else:
                resampling = Resampling.bilinear
            
            out_array = np.zeros((dst_height, dst_width), dtype=dtype)
            reproject(
                source=band['data'],
                destination=out_array,
                src_transform=band['transform'],
                src_crs=band['crs'],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=nodata_value,
                dst_nodata=nodata_value
            )
            transformed.append(out_array)
        
        arrays = transformed
        src_width, src_height = dst_width, dst_height
        src_transform = dst_transform

    profile = {
        'driver': 'GTiff',
        'count': len(band_names),
        'dtype': dtype,
        'width': src_width,
        'height': src_height,
        'crs': dst_crs,
        'transform': src_transform,
        'compress': compress,
        'nodata': nodata_value
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, (array, name) in enumerate(zip(arrays, band_names), 1):
            dst.write(array, indexes=i)
            dst.set_band_description(i, name)
    
    print(f"  Сохранено: {output_path}")
    return output_path