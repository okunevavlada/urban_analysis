"""
Функция расчета параметров подстилающей поверхности
"""
from typing import Union, List, Optional

def reproject_resample_clip(
    input_rasters: Union[str, List[str]],
    reference_raster: str,
    output_path: str,
    crs_code: str = 'EPSG:32645',
    resolution: float = 2.0,
    resampling: str = 'nearest',
    band_number: Optional[int] = None  
) -> str:
    """
    Объединение, перепроецирование, обрезка и изменение разрешения растра

    Args:
        input_rasters: путь / список / паттерн
        reference_raster: растр для обрезки
        output_path: выходной файл
        crs_code: целевая проекция
        resolution: целевое разрешение
        resampling: метод ресэмплинга
        band_number: номер канала 

    Returns:
        путь к результату
    """
    import os
    import glob
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask
    from rasterio.merge import merge

    if isinstance(input_rasters, list):
        input_rasters = input_rasters
    elif isinstance(input_rasters, str):
        if '*' in input_rasters or '?' in input_rasters:
            input_rasters = glob.glob(input_rasters)
        else:
            input_rasters = [input_rasters]

    for r in input_rasters:
        print(f"      - {os.path.basename(r)}")

    resample_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }
    resample_method = resample_map.get(resampling, Resampling.nearest)

    temp_dir = os.path.dirname(output_path)
    temp_mosaic = os.path.join(temp_dir, '_temp_mosaic.tif')
    temp_reprojected = os.path.join(temp_dir, '_temp_reprojected.tif')
    temp_clipped = os.path.join(temp_dir, '_temp_clipped.tif')

    if len(input_rasters) == 1:
        with rasterio.open(input_rasters[0]) as src:
            profile = src.meta.copy()

            if band_number is not None:
                data = src.read(band_number)
                data = data[np.newaxis, ...] 

                profile.update({'count': 1})
            else:
                data = src.read() 

            with rasterio.open(temp_mosaic, 'w', **profile) as dst:
                dst.write(data)

    else:
        src_files = [rasterio.open(r) for r in input_rasters]

        mosaic, out_transform = merge(src_files)

        out_meta = src_files[0].meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform,
            'count': mosaic.shape[0]
        })

        with rasterio.open(temp_mosaic, 'w', **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files:
            src.close()

    with rasterio.open(temp_mosaic) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs_code, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_code,
            'transform': transform,
            'width': width,
            'height': height,
            'count': src.count
        })

        with rasterio.open(temp_reprojected, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_code,
                    resampling=resample_method
                )

    with rasterio.open(temp_reprojected) as src, rasterio.open(reference_raster) as ref:
        bbox = [{
            'type': 'Polygon',
            'coordinates': [[
                [ref.bounds.left, ref.bounds.bottom],
                [ref.bounds.left, ref.bounds.top],
                [ref.bounds.right, ref.bounds.top],
                [ref.bounds.right, ref.bounds.bottom],
                [ref.bounds.left, ref.bounds.bottom]
            ]]
        }]

        out_image, out_transform = mask(src, bbox, crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })

        with rasterio.open(temp_clipped, 'w', **out_meta) as dst:
            dst.write(out_image)

    with rasterio.open(temp_clipped) as dataset:
        current_res_x = dataset.transform.a
        current_res_y = abs(dataset.transform.e)

        scale_x = current_res_x / resolution
        scale_y = current_res_y / resolution

        new_height = int(dataset.height * scale_y)
        new_width = int(dataset.width * scale_x)

        data = dataset.read(
            out_shape=(dataset.count, new_height, new_width),
            resampling=resample_method
        )

        out_meta = dataset.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': new_height,
            'width': new_width,
            'transform': dataset.transform * dataset.transform.scale(
                1 / scale_x,
                1 / scale_y
            )
        })

        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(data)

    if os.path.exists(temp_mosaic):
        os.remove(temp_mosaic)
    if os.path.exists(temp_reprojected):
        os.remove(temp_reprojected)
    if os.path.exists(temp_clipped):
        os.remove(temp_clipped)

    return output_path