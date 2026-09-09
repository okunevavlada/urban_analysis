"""
Модуль для полной обработки растров зданий:
1. Расчет евклидова расстояния
2. Расчет аллокации
3. Расчет тайлов ширины
4. Создание выходного растра
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.merge import merge

import glob
from pathlib import Path
from typing import Optional, Union, Dict, Tuple


def process_building_rasters(
    input_raster: str,
    output_dir: str,
    resolution: float = 2.0,
    window_size: tuple = (2, 3),
    hard: bool = False,
    n_output_bands: int = 3,

    distance_name: str = "distance.tif",
    allocation_name: str = "allocation.tif",
    tiles_name: str = "tiles_width.npy",
    output_name: str = "canyon_params.tif",

    use_pygeos: bool = False,
    dtype: str = "int16",
    compress: str = "lzw"
 ) -> dict:
    """
    Подготовка данных для расчета параметров городских каньонов:
    1. Евклидово расстояние
    2. Аллокация
    3. Тайлы ширины
    4. Пустой многоканальный выходной растр
    """

    import os
    import numpy as np
    import rasterio
    import rasterspace as rs

    if not use_pygeos:
        os.environ["USE_PYGEOS"] = "0"

    os.makedirs(output_dir, exist_ok=True)

    distance_path = os.path.join(output_dir, distance_name)
    allocation_path = os.path.join(output_dir, allocation_name)
    tiles_path = os.path.join(output_dir, tiles_name)
    output_path = os.path.join(output_dir, output_name)

    # Чтение входного растра
    with rasterio.open(input_raster) as src:
        array = src.read(1).astype("float64")
        profile = src.profile.copy()

    # Евклидово расстояние
    euc = rs.euclidean_distance(array, resolution)

    dist_profile = profile.copy()
    dist_profile.pop("blockxsize", None)
    dist_profile.pop("blockysize", None)
    dist_profile.update(
        count=1,
        dtype="float64",
        compress=compress
    )

    with rasterio.open(distance_path, "w", **dist_profile) as dst:
        dst.write(euc, 1)

    # Аллокация
    alloc = rs.euclidean_allocation(array, resolution)

    alloc_profile = profile.copy()
    alloc_profile.pop("blockxsize", None)
    alloc_profile.pop("blockysize", None)
    alloc_profile.update(
        count=1,
        dtype="float64",
        compress=compress
    )

    with rasterio.open(allocation_path, "w", **alloc_profile) as dst:
        dst.write(alloc, 1)

    # Тайлы ширины
    tiles_width = rs.euclidean_width_tiles(
        euc,
        resolution,
        window_size[0],
        window_size[1],
        hard
    )

    with open(tiles_path, "wb") as dst:
        np.save(dst, tiles_width)

    # Пустой многоканальный растр
    out_profile = profile.copy()
    out_profile.pop("blockxsize", None)
    out_profile.pop("blockysize", None)

    out_profile.update(
        count=n_output_bands,
        dtype=dtype,
        compress=compress,
        tiled=True,
        BIGTIFF="YES"
    )

    with rasterio.open(output_path, "w", **out_profile):
        pass

    # Результат
    return {
        "input_raster": input_raster,
        "distance_raster": distance_path,
        "allocation_raster": allocation_path,
        "tiles_file": tiles_path,
        "output_raster": output_path
    }


# ФУНКЦИИ ДЛЯ ОТДЕЛЬНЫХ ЭТАПОВ 

def calculate_distance_only(
    input_raster: str,
    output_path: str,
    resolution: float = 2.0
) -> str:
    """Только расчет евклидова расстояния"""
    import rasterspace as rs
    with rasterio.open(input_raster) as src:
        array = src.read(1).astype('float64')
        profile = src.profile
    
    euc = rs.euclidean_distance(array, resolution)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(euc, indexes=1)
    
    return output_path


def calculate_allocation_only(
    input_raster: str,
    output_path: str,
    resolution: float = 2.0
) -> str:
    """Только расчет аллокации"""
    import rasterspace as rs
    with rasterio.open(input_raster) as src:
        array = src.read(1).astype('float64')
        profile = src.profile
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    
    alloc = rs.euclidean_allocation(array, resolution)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(alloc, indexes=1)
    
    return output_path


def calculate_width_tiles_only(
    distance_raster: str,
    output_path: str,
    resolution: float = 2.0,
    window_size: Tuple[int, int] = (2, 3),
    hard: bool = False
) -> str:
    """Только расчет тайлов ширины"""
    import rasterspace as rs
    with rasterio.open(distance_raster) as src:
        euc = src.read(1).astype('float64')
    
    tiles_width = rs.euclidean_width_tiles(
        euc, resolution, 
        window_size[0], window_size[1], 
        hard
    )
    
    with open(output_path, 'wb') as dst:
        np.save(dst, tiles_width)
    
    return output_path

def create_empty_multiband_raster(
    output_path: str,
    template_raster: str,
    n_bands: int = 3,
    dtype: str = 'int16',
    compress: str = 'lzw',
    nodata_value: int = -1
) -> str:

    import os
    import rasterio

    with rasterio.open(template_raster) as src:
        profile = src.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        count=n_bands,
        dtype=dtype,
        compress=compress,
        tiled=True,
        BIGTIFF='YES',
        nodata=nodata_value
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **profile):
        pass

    return output_path

