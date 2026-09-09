# def process_canyon_length(
#     tiles_path: str,
#     output_path: str,
#     distance_path: str,
#     allocation_path: str,
#     width_path: str,
#     target_tiles: list = None,
#     resolution: float = 2.0,
#     max_radius: float = 1000.0,
#     direction: float = 5.0,
#     use_pygeos: bool = False,
#     template_raster: str = None,
#     n_bands: int = 4,
#     dtype: str = 'float32',
#     compress: str = 'lzw',
#     nodata_value: int = -1
# ) -> dict:
#     """
#     Обработка параметров длины каньонов 

#      Args:
#         tiles_path: файл с тайлами (.npy)
#         output_path: выходной растр
#         distance_path: растр  евклидова расстояния до зданий
#         allocation_path: растр аллокации (распределения высот)
#         width_path: растр ширины городских каньонов
#         target_tiles: список тайлов для обработки [(i, j), ...]; если None, обрабатываются все тайлы
#         resolution: разрешение растра 
#         max_radius: максимальный радиус поиска в метрах
#         direction: дискретность по углу в градусах (допустимые значения: 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90); по умолчанию 5.0
#         use_pygeos: использовать ли PYGEOS 
#         template_raster: выходной файл
#         n_bands: количество каналов в выходном растре 
#         dtype: тип данных выходного растра
#         compress: метод сжатия 
#         nodata_value: значение NoData для выходного растра
        
#     """
#     import os
#     import numpy as np
#     import rasterio
#     from rasterio.windows import Window
#     import rasterspace as rs
    
#     if not use_pygeos:
#         os.environ['USE_PYGEOS'] = '0'
    
#     tiles = np.load(tiles_path)

#     if not os.path.exists(output_path):
#         if template_raster:
#             from urban_analysis import create_empty_multiband_raster
            
#             create_empty_multiband_raster(
#                 output_path=output_path,
#                 template_raster=template_raster,
#                 n_bands=n_bands,
#                 dtype=dtype,
#                 compress=compress,
#                 nodata_value=nodata_value
#             )
#         else:
#             raise FileNotFoundError(f"Файл {output_path} не существует")
#     else:
#         print(f"Продолжение записи")
    
#     if target_tiles is None:
#         target_tiles = []
#         for i in range(tiles.shape[0]):
#             for j in range(tiles.shape[1]):
#                 target_tiles.append((i, j))
    
#     processed_count = 0
    
#     with rasterio.open(output_path, 'r+') as dst:
#         for idx, (i, j) in enumerate(target_tiles):
#             tile_id = f'TILE {i}, {j}'
#             print(f'\n Обработка {tile_id} ({idx+1}/{len(target_tiles)})')
            
#             win_write = Window.from_slices(
#                 (tiles[i, j, 0], tiles[i, j, 1]),
#                 (tiles[i, j, 2], tiles[i, j, 3])
#             )
#             win_read = Window.from_slices(
#                 (tiles[i, j, 4], tiles[i, j, 5]),
#                 (tiles[i, j, 6], tiles[i, j, 7])
#             )
            
#             with rasterio.open(distance_path) as src_d:
#                 distance = src_d.read(1, window=win_read).astype(np.float64)
            
#             with rasterio.open(allocation_path) as src_a:
#                 allocation = src_a.read(1, window=win_read).astype(np.float64)
            
#             with rasterio.open(width_path) as src_w:
#                 width = src_w.read(1, window=win_read).astype(np.float64)
            
#             params = rs.euclidean_length_params(
#                 distance, allocation, width, 
#                 direction, max_radius, resolution
#             )
            
#             row1 = tiles[i, j, 0] - tiles[i, j, 4]
#             col1 = tiles[i, j, 2] - tiles[i, j, 6]
#             height = int(win_write.height)
#             width_win = int(win_write.width)
            
#             for k in range(3):
#                 band = k + 2
#                 write_data = params[k, row1:row1 + height, col1:col1 + width_win]
#                 dst.write(write_data, indexes=band, window=win_write)
            
#             processed_count += 1
    
#     return {
#         'processed_tiles': processed_count,
#         'target_tiles': target_tiles,
#         'output_file': output_path,
#         'status': 'success'
#     }

def process_canyon_length(
    tiles_path: str,
    output_path: str,
    distance_path: str,
    allocation_path: str,
    width_path: str,
    target_tiles: list = None,
    resolution: float = 2.0,
    max_radius: float = 1000.0,
    direction: float = 5.0,
    use_pygeos: bool = False,
    template_raster: str = None,
    n_bands: int = 4,
    dtype: str = 'float64',
    compress: str = 'lzw',
    nodata_value: int = -1
) -> dict:
    """
    Обработка параметров длины каньонов 
    """
    import os
    import gc
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import rasterspace as rs
    
    if not use_pygeos:
        os.environ['USE_PYGEOS'] = '0'
    
    tiles = np.load(tiles_path)
    
    # Создание выходного файла если не существует
    if not os.path.exists(output_path):
        if template_raster:
            from urban_analysis import create_empty_multiband_raster
            
            create_empty_multiband_raster(
                output_path=output_path,
                template_raster=template_raster,
                n_bands=n_bands,
                dtype=dtype,
                compress=compress,
                nodata_value=nodata_value
            )
        else:
            raise FileNotFoundError(f"Файл {output_path} не существует")
    else:
        print(f"Продолжение записи")
    
    if target_tiles is None:
        target_tiles = []
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                target_tiles.append((i, j))
    
    processed_count = 0
    
    # Открываем все исходные растры один раз за пределами цикла
    with rasterio.open(distance_path) as src_d, \
         rasterio.open(allocation_path) as src_a, \
         rasterio.open(width_path) as src_w, \
         rasterio.open(output_path, 'r+') as dst:
        
        for idx, (i, j) in enumerate(target_tiles):
            tile_id = f'TILE {i}, {j}'
            print(f'\nОбработка {tile_id} ({idx+1}/{len(target_tiles)})')
            
            # Получаем окна из тайлов
            win_write = Window.from_slices(
                (tiles[i, j, 0], tiles[i, j, 1]),
                (tiles[i, j, 2], tiles[i, j, 3])
            )
            win_read = Window.from_slices(
                (tiles[i, j, 4], tiles[i, j, 5]),
                (tiles[i, j, 6], tiles[i, j, 7])
            )
            
            # Читаем данные окон
            distance = src_d.read(1, window=win_read).astype(np.float64)
            allocation = src_a.read(1, window=win_read).astype(np.float64)
            width = src_w.read(1, window=win_read).astype(np.float64)
            
            # Вычисление параметров
            params = rs.euclidean_length_params(
                distance, allocation, width, 
                direction, max_radius, resolution
            )

            row_start = tiles[i, j, 0] - tiles[i, j, 4]
            col_start = tiles[i, j, 2] - tiles[i, j, 6]
            height = int(win_write.height)
            width_win = int(win_write.width)
            
            # Запись в выходной растр
            for k in range(3):
                band = k + 2
                write_data = params[k, row_start:row_start + height, col_start:col_start + width_win]
                dst.write(write_data, indexes=band, window=win_write)
            
            processed_count += 1
            
   
    return {
        'processed_tiles': processed_count,
        'target_tiles': target_tiles,
        'output_file': output_path,
        'status': 'success'
    }


