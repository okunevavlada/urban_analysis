"""
Функция расчета промышленных зон 
"""

def rasterize_industrial(
    industrial_path: str,
    building_raster_path: str,
    output_path: str,
    target_value: str = 'industrial',
    resolution: float = 5.0,
    crs_code: int = 32645
) -> str:
    """
    Растеризация промышленных зон
    
    Args:
        industrial_path: полный путь к файлу с промышленными зонами в формате GeoJSON
        building_raster_path: полный путь к растру зданий (для обрезки по границам)
        output_path: полный путь для сохранения выходного растра
        target_value: значение класса для растеризации (по умолчанию 'industrial')
        resolution: разрешение выходного растра в метрах (по умолчанию 5.0)
        crs_code: код проекции в EPSG (по умолчанию 32645)
    
    Returns:
        str: путь к сохраненному растру
    """
    import os
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box
    
    # Загрузка промышленных зон
    industri = gpd.read_file(industrial_path)
    industri = industri.to_crs(epsg=crs_code)
    
    # Получение границ из растра зданий
    with rasterio.open(building_raster_path) as src:
        bounds = src.bounds
    
    # Обрезка по границам растра
    polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    clip_polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs=industri.crs)
    industri_clip = gpd.clip(industri, clip_polygon)
    
    # Создание бинарной маски
    industri_clip['raster_value'] = np.where(industri_clip['class'] == target_value, 1, 0)
    
    # Расчет размеров растра
    bounds_clip = industri_clip.total_bounds
    width = int((bounds_clip[2] - bounds_clip[0]) / resolution)
    height = int((bounds_clip[3] - bounds_clip[1]) / resolution)
    transform = rasterio.transform.from_origin(bounds_clip[0], bounds_clip[3], resolution, resolution)
    
    # Растеризация
    raster = rasterize(
        [(geom, value) for geom, value in zip(industri_clip.geometry, industri_clip['raster_value'])],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        compress='lzw',
        BIGTIFF='YES',
        dtype='uint8',
        crs=industri.crs,
        transform=transform
    ) as dst:
        dst.write(raster, 1)
    
    
    return output_path