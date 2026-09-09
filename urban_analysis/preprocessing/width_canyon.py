"""
Расчет ширины городского каньона 
"""

def canyon_width_tiles(
    wd: str = 'msk_data',
    tiles_file: str = 'tiles_width_msk_correct.npy',
    distance_file: str = 'distance_msk.tif',
    allocation_file: str = 'allocation_msk.tif',
    output_file: str = 'canyon_params_msk.tif',
    target_tiles: list = None,
    skip_existing: bool = True,
    resolution: float = 5.0,
    window_size: tuple = (2, 2),
    max_radius: float = 500.0,
    use_absolute: bool = True
) -> dict:
    """
    Расчет ширины каньона по тайлам
    
    Args:
        wd: рабочая директория
        tiles_file: имя файла с тайлами (.npy)
        distance_file: имя файла растра расстояний
        allocation_file: имя файла растра аллокации
        output_file: имя выходного файла
        target_tiles: список тайлов для обработки [(i,j), ...] (если None - все)
        skip_existing: пропускать уже обработанные тайлы
        resolution: разрешение (по умолчанию 5.0)
        window_size: размер окна (height, width) (по умолчанию (2, 2))
        max_radius: максимальный радиус (по умолчанию 500.0)
        use_absolute: использовать абсолютные значения (по умолчанию True)
    
    Returns:
        dict: словарь с результатами обработки
    """
    import os
    import rasterio
    from rasterio.windows import Window
    import rasterspace as rs
    import numpy as np

    os.environ['USE_PYGEOS'] = '0'

    tiles_path = os.path.join(wd, tiles_file)
    distance_path = os.path.join(wd, distance_file)
    allocation_path = os.path.join(wd, allocation_file)
    output_path = os.path.join(wd, output_file)

    tiles = np.load(tiles_path)

    if target_tiles is None:
        target_tiles = []
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                target_tiles.append((i, j))
    
    processed_count = 0
    skipped_count = 0

    window_height, window_width = window_size
    
    with rasterio.open(output_path, 'r+') as dst:
        for idx, (i, j) in enumerate(target_tiles):
            tile_id = f'TILE {i}, {j}'
            
            # Проверка, обработан ли уже тайл
            if skip_existing:
                win_write = Window.from_slices(
                    (tiles[i, j, 0], tiles[i, j, 1]),
                    (tiles[i, j, 2], tiles[i, j, 3])
                )
                
                # Читаем второй канал для проверки
                existing_data = dst.read(2, window=win_write)
                
                # Если в тайле есть данные (не все нули), пропускаем
                if existing_data is not None and np.any(existing_data != 0):
                    print(f' Пропуск {tile_id} ({idx+1}/{len(target_tiles)}) - уже обработан')
                    skipped_count += 1
                    continue
            
            print(f'➡ Обработка {tile_id} ({idx+1}/{len(target_tiles)})')
            
            win_write = Window.from_slices(
                (tiles[i, j, 0], tiles[i, j, 1]),
                (tiles[i, j, 2], tiles[i, j, 3])
            )
            win_read = Window.from_slices(
                (tiles[i, j, 4], tiles[i, j, 5]),
                (tiles[i, j, 6], tiles[i, j, 7])
            )
            
            with rasterio.open(distance_path) as src_d:
                distance = src_d.read(1, window=win_read).astype(np.float64)
            with rasterio.open(allocation_path) as src_a:
                allocation = src_a.read(1, window=win_read).astype(np.float64)
            
            params = rs.euclidean_width_params_split(
                distance, allocation, resolution, 
                window_height, window_width, use_absolute, max_radius
            )
            
            row1 = tiles[i, j, 0] - tiles[i, j, 4]
            row2 = tiles[i, j, 1] - tiles[i, j, 4]
            col1 = tiles[i, j, 2] - tiles[i, j, 6]
            col2 = tiles[i, j, 3] - tiles[i, j, 6]
            
            for k in range(2):
                band = k + 2
                write_data = params[k, row1:row2, col1:col2]
                dst.write(write_data, band, window=win_write)
            
            processed_count += 1
    
    return {
        'processed_tiles': processed_count,
        'skipped_tiles': skipped_count,
        'target_tiles': target_tiles,
        'output_file': output_path,
        'status': 'success'
    }