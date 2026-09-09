# Автоматизированный расчет параметров 

import os
import glob 
import pickle
import yaml
import logging
import sys
import json
import argparse
from datetime import datetime
from urban_analysis import (
    rasterize_buildings,
    process_building_rasters,
    canyon_width_tiles,
    process_width_raster_pipeline,
    dem_full_pipeline,
    lcz_prepair,
    calculate_building_parameters,
    restore_building_height_model,
    find_best_catboost_params,
    predict_with_trained_model,
    rasterize_both_heights,
    create_empty_multiband_raster, 
    calculate_allocation_only, 
    process_canyon_length,
    rasterize_roads,
    reproject_resample_clip,
    rasterize_industrial,
    create_multiband_raster
)

# НАСТРОЙКА ФАЙЛА СОСТОЯНИЯ

class PipelineState:

    def __init__(self, config_path: str, output_dir: str):
        self.config_path = config_path
        self.output_dir = output_dir
        self.current_step = 0
        self.completed_steps = set()
        self.state = {}
        
    def _get_state_file(self):
        return os.path.join(self.output_dir, 'pipeline_state.pkl')
    
    def save(self, step: int, data: dict):
        if step > self.current_step:
            self.current_step = step

        self.completed_steps.add(step)

        data[f'_step_{step}_time'] = datetime.now().isoformat()
        data[f'_step_{step}_completed'] = True

        self.state.update(data)
        
        state_file = self._get_state_file()
        with open(state_file, 'wb') as f:
            pickle.dump({
                'current_step': self.current_step,
                'completed_steps': list(self.completed_steps),
                'state': self.state,
                'config_path': self.config_path,
                'timestamp': datetime.now().isoformat()
            }, f)
        print(f"Состояние сохранено (шаг {step})")
    
    def load(self) -> int:
        state_file = self._get_state_file()
        
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                data = pickle.load(f)
            
            self.current_step = data['current_step']
            self.completed_steps = set(data.get('completed_steps', []))
            self.state = data['state']
            
            print(f"Загружено состояние (шаг {self.current_step}, выполнено: {sorted(self.completed_steps)})")
            return self.current_step
        
        return 0
    
    def is_step_completed(self, step: int) -> bool:
        return step in self.completed_steps
    
    def get_completed_steps(self) -> list:
        return sorted(self.completed_steps)
    
    def get(self, key: str, default=None):
        return self.state.get(key, default)
    
    def get_step_time(self, step: int) -> str:
        return self.state.get(f'_step_{step}_time', 'время не сохранено')

# НАСТРОЙКА ЛОГГИРОВАНИЯ 

def setup_logging(log_dir: str = None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'pipeline.log')
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=log_datefmt,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

# ЗАГРУЗКА КОНФИГУРАЦИОННОГО ФАЙЛА

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ФУНКЦИЯ РАСЧЕТА ПАРАМЕТРОВ 

def run_pipeline(
    config_path: str = "config.yaml",
    start_from: int = None,
    end_step: int = None,
    single_step: bool = False
):
    """
    Args:
        config_path: путь к конфигурационному файлу
        start_from: номер шага, с которого начать
        end_step: номер шага, на котором остановиться
        single_step: выполнить только один шаг
    """
    
    config = load_config(config_path)

    WD = config['wd']
    OUTPUT_DIR = os.path.join(WD, config['output_dir'])
    MODELS_DIR = os.path.join(WD, config['models_dir'])
    LOGS_DIR = os.path.join(WD, config.get('logs_dir', 'logs'))
    READY_DIR = os.path.join(WD, config.get('ready_dir', 'ready_base'))
    
    CRS_CODE = config['crs_code']
    RESOLUTION = config['resolution']

    MAX_STEP = 19
    if end_step is None:
        end_step = MAX_STEP
    end_step = min(end_step, MAX_STEP)

    state = PipelineState(config_path, OUTPUT_DIR)
    logger = setup_logging(LOGS_DIR)

    state.load()
    completed_steps = state.get_completed_steps()

    if start_from is None:
        start_from = 1
        while start_from <= MAX_STEP and state.is_step_completed(start_from):
            start_from += 1

    print(f"Старт с шага {start_from}")
    print(f"Выполненные шаги: {completed_steps}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(READY_DIR, exist_ok=True)
    
    input_cfg = config['input']
    output_cfg = config['output']
    proc_cfg = config['processing']
    bands_cfg = config.get('bands', {})

    final = None
    current_data = {}

    # ШАГ 1
    if start_from <= 1 <= end_step:
        logger.info("ШАГ 1: Растеризация зданий")
        
        buildings_raster = rasterize_buildings(
            buildings_path=os.path.join(WD, input_cfg['buildings_geojson']),
            output_raster=os.path.join(OUTPUT_DIR, output_cfg['buildings_raster']),
            resolution=RESOLUTION,
            crs_code=CRS_CODE,
            fill_value=proc_cfg['buildings']['fill_value']
        )

        state.save(1, {
            'buildings_raster': buildings_raster,
            'output_dir': OUTPUT_DIR
        })
        current_data['buildings_raster'] = buildings_raster
        logger.info(f"Сохранено: {buildings_raster}")

        if single_step and start_from == 1:
            return final
        if end_step is not None and 1 >= end_step:
            return final

    else:
        if state.is_step_completed(1):
            buildings_raster = state.get('buildings_raster')
            current_data['buildings_raster'] = buildings_raster
            logger.info(f"Шаг 1 пропущен, используем сохраненный")
        else:
            buildings_raster = None
            logger.info("Шаг 1 пропущен")

    # ШАГ 2
    preproc_cfg = proc_cfg['preprocessing']

    if start_from <= 2 <= end_step:
        logger.info("ШАГ 2: Препроцессинг")
        
        preproc_result = process_building_rasters(
            input_raster=buildings_raster,
            output_dir=OUTPUT_DIR,
            resolution=RESOLUTION,
            window_size=preproc_cfg['window_size'],
            hard=preproc_cfg['hard'],
            n_output_bands=3,
            distance_name=output_cfg['distance'],
            allocation_name=output_cfg['allocation'],
            tiles_name=output_cfg['tiles_width'],
            output_name=output_cfg['canyon_params_raw']
        )
        
        state.save(2, {
            'distance': preproc_result['distance_raster'],
            'allocation': preproc_result['allocation_raster'],
            'tiles': preproc_result['tiles_file']
        })
        current_data['distance'] = preproc_result['distance_raster']
        current_data['allocation'] = preproc_result['allocation_raster']
        current_data['tiles'] = preproc_result['tiles_file']
        logger.info("Шаг 2: выполнен")

        if single_step and start_from == 2:
            return final
        if end_step is not None and 2 >= end_step:
            return final

    else:
        if state.is_step_completed(2):
            distance = state.get('distance')
            allocation = state.get('allocation')
            tiles = state.get('tiles')
            if distance:
                current_data['distance'] = distance
            if allocation:
                current_data['allocation'] = allocation
            if tiles:
                current_data['tiles'] = tiles
            logger.info("Шаг 2 пропущен, используем сохраненные")
        else:
            distance = None
            allocation = None
            tiles = None
            logger.info("Шаг 2 пропущен")

    # ШАГ 3
    if start_from <= 3 <= end_step:
        logger.info("ШАГ 3: Расчет ширины каньона по тайлам (первый)")
        
        wc_cfg = proc_cfg['canyon_width_tiles_first']
        window_size = preproc_cfg['window_size']
        max_radius = wc_cfg.get('max_radius', 500.0)
        
        width_result = canyon_width_tiles(
            wd=OUTPUT_DIR, 
            tiles_file=output_cfg['tiles_width'],
            distance_file=output_cfg['distance'],
            allocation_file=output_cfg['allocation'],
            output_file=output_cfg['canyon_params_raw'], 
            target_tiles=wc_cfg.get('target_tiles'),
            skip_existing=wc_cfg.get('skip_existing', True),
            resolution=RESOLUTION,
            window_size=tuple(window_size),
            max_radius=max_radius,
            use_absolute=wc_cfg.get('use_absolute', True)
        )
        
        state.save(3, {'canyon_params_raw': width_result['output_file']})
        current_data['canyon_params_raw'] = width_result['output_file']
        logger.info("Шаг 3: выполнен")

        if single_step and start_from == 3:
            return final
        if end_step is not None and 3 >= end_step:
            return final

    else:
        if state.is_step_completed(3):
            canyon_params_raw = state.get('canyon_params_raw')
            if canyon_params_raw:
                current_data['canyon_params_raw'] = canyon_params_raw
            logger.info("Шаг 3 пропущен, используем сохраненный")
        else:
            canyon_params_raw = None
            logger.info("Шаг 3 пропущен")
    
    # ШАГ 4
    if start_from <= 4 <= end_step:
        logger.info("ШАГ 4: Фильтрация ширины и извлечение статистики")
        
        width_cfg = proc_cfg['width']
        band = bands_cfg.get('canyon_width', width_cfg.get('band', 2))
        
        width_raster = current_data.get('canyon_params_raw') or state.get('canyon_params_raw')
        if width_raster is None:
            width_raster = os.path.join(OUTPUT_DIR, output_cfg.get('canyon_params_raw', 'canyon_params_raw.tif'))
        
        width_filtered = process_width_raster_pipeline(
            width_raster=width_raster,
            polygons=os.path.join(WD, input_cfg['buildings_geojson']),
            output_dir=OUTPUT_DIR,
            band=band,
            clip_raster_name=output_cfg.get('width_raster', 'width_clip.tif'),
            filled_raster_name=output_cfg.get('width_filled', 'width_filled.tif'),
            filtered_raster_name=output_cfg.get('width_filtered', 'width_filtered.tif'),
            output_csv_name=output_cfg['width_csv'],
            invert_mask=width_cfg['invert_mask'],
            nodata_value=width_cfg['nodata_value'],
            fill_filter_size=width_cfg['fill_filter_size'],
            mean_filter_size=width_cfg['mean_filter_size'],
            zonal_stats_list=width_cfg['zonal_stats'],
            fallback_to_centroid=width_cfg['fallback_to_centroid']
        )
        
        state.save(4, {'width_csv': width_filtered['output_csv']})
        current_data['width_csv'] = width_filtered['output_csv']
        logger.info("Шаг 4: выполнен")

        if single_step and start_from == 4:
            return final
        if end_step is not None and 4 >= end_step:
            return final

    else:
        if state.is_step_completed(4):
            width_csv = state.get('width_csv')
            if width_csv:
                current_data['width_csv'] = width_csv
            logger.info("Шаг 4 пропущен, используем сохраненный")
        else:
            width_csv = None
            logger.info("Шаг 4 пропущен")

    # ШАГ 5
    if start_from <= 5 <= end_step:
        logger.info("ШАГ 5: Обработка DEM")

        dem_type = proc_cfg.get('dem_type', 'fabdem')
        dem_cfg = proc_cfg.get(dem_type, {})

        if dem_type == 'arcticdem':
            dem_result = dem_full_pipeline(
                roads_path=os.path.join(WD, input_cfg['roads_geojson']),
                buildings_path=os.path.join(WD, input_cfg['buildings_geojson']),
                dem_tiles_pattern=os.path.join(WD, input_cfg['dem_tiles_pattern']),
                dem_type='arcticdem',
                output_dir=OUTPUT_DIR,
                output_csv=os.path.join(OUTPUT_DIR, output_cfg['dem_csv']),
                use_difference=dem_cfg.get('use_difference', True),
                buffer_degrees=dem_cfg.get('buffer_degrees', 0.005),
                nodata_value=dem_cfg.get('nodata_value', -9999.0),
                chunk_size_interp=dem_cfg.get('chunk_size_interp', 500),
                chunk_size_io=dem_cfg.get('chunk_size_io', 1000),
                use_tqdm=dem_cfg.get('use_tqdm', True)
            )
            dem_csv_val = dem_result['buildings_csv']
            state.save(5, {'dem_csv': dem_csv_val})
            current_data['dem_csv'] = dem_csv_val

        elif dem_type == 'fabdem':
            dem_result = dem_full_pipeline(
                roads_path=os.path.join(WD, input_cfg['roads_geojson']),
                buildings_path=os.path.join(WD, input_cfg['buildings_geojson']),
                fabdem_tiles_pattern=os.path.join(WD, input_cfg['fabdem_tiles_pattern']),
                copernicus_dem=os.path.join(WD, input_cfg['copernicus_dem']),
                dem_type='fabdem',
                output_dir=OUTPUT_DIR,
                output_csv=os.path.join(OUTPUT_DIR, output_cfg['dem_csv']),
                use_difference=dem_cfg.get('use_difference', True),
                nodata_value=dem_cfg.get('nodata_value', -9999.0),
                chunk_size_io=dem_cfg.get('chunk_size_io', 1000),
                use_tqdm=dem_cfg.get('use_tqdm', True)
            )
            dem_csv_val = dem_result['buildings_csv']
            state.save(5, {
                'dem_csv': dem_csv_val,
                'fabdem_original': os.path.join(OUTPUT_DIR, "fabdem_merged_original.tif")
            })
            current_data['dem_csv'] = dem_csv_val

        logger.info("Шаг 5: выполнен")

        if single_step and start_from == 5:
            return final
        if end_step is not None and 5 >= end_step:
            return final

    else:
        if state.is_step_completed(5):
            dem_csv = state.get('dem_csv')
            if dem_csv:
                current_data['dem_csv'] = dem_csv
            logger.info("Шаг 5 пропущен, используем сохраненный")
        else:
            dem_csv = None
            logger.info("Шаг 5 пропущен")
    
    # ШАГ 6
    if start_from <= 6 <= end_step:
        logger.info("ШАГ 6: LCZ")

        lcz_cfg = proc_cfg['lcz']

        lcz_result = lcz_prepair(
            lcz_raster=os.path.join(WD, input_cfg['lcz_raster']),
            buildings_path=os.path.join(WD, input_cfg['buildings_geojson']),
            output_dir=OUTPUT_DIR,
            remove_values=lcz_cfg.get('remove_values', [11, 12, 13, 14, 15, 16, 17]),
            fill_filter_size=lcz_cfg.get('fill_filter_size', 3),
            buffer_meters=lcz_cfg.get('buffer_meters', 100),
            processed_raster_name=output_cfg.get('lcz_processed'),
            output_csv_name=output_cfg['lcz_csv'],
            band_number=lcz_cfg.get('band_number', 2),
            crs_code=CRS_CODE
        )

        lcz_csv_val = lcz_result['output_csv']
        state.save(6, {
            'lcz_csv': lcz_csv_val,
            'lcz_processed': lcz_result['processed_raster']
        })
        current_data['lcz_csv'] = lcz_csv_val
        logger.info("Шаг 6: выполнен")

        if single_step and start_from == 6:
            return final
        if end_step is not None and 6 >= end_step:
            return final

    else:
        if state.is_step_completed(6):
            lcz_csv = state.get('lcz_csv')
            if lcz_csv:
                current_data['lcz_csv'] = lcz_csv
            logger.info("Шаг 6 пропущен, используем сохраненный")
        else:
            lcz_csv = None
            logger.info("Шаг 6 пропущен")
    
    # ШАГ 7
    if start_from <= 7 <= end_step:
        logger.info("ШАГ 7: подготовка данных")

        calculate_building_parameters(
            buildings_geojson=os.path.join(WD, input_cfg['buildings_geojson']),
            width_csv=os.path.join(OUTPUT_DIR, output_cfg['width_csv']),
            dem_csv=os.path.join(OUTPUT_DIR, output_cfg['dem_csv']),
            lcz_csv=os.path.join(OUTPUT_DIR, output_cfg['lcz_csv']),
            output_csv=os.path.join(OUTPUT_DIR, output_cfg['prepared_csv']),
            crs_code=CRS_CODE
        )

        prepared_csv_path = os.path.join(OUTPUT_DIR, output_cfg['prepared_csv'])
        state.save(7, {'prepared_csv': prepared_csv_path})
        current_data['prepared_csv'] = prepared_csv_path
        logger.info("Шаг 7: выполнен")

        if single_step and start_from == 7:
            return final
        if end_step is not None and 7 >= end_step:
            return final

    else:
        if state.is_step_completed(7):
            prepared_csv = state.get('prepared_csv')
            if prepared_csv and isinstance(prepared_csv, str):
                current_data['prepared_csv'] = prepared_csv
                logger.info("Шаг 7 пропущен, используем сохраненный")
            else:
                logger.warning("Шаг 7: prepared_csv имеет неверный тип")
        else:
            prepared_csv = None
            logger.info("Шаг 7 пропущен")

    if start_from <= 8 <= end_step:
        logger.info("ШАГ 8: обучение модели")

        ml_train_cfg = proc_cfg['ml_train']
        ml_predict_cfg = proc_cfg['ml_predict']
        
        prepared_csv_path = current_data.get('prepared_csv') or state.get('prepared_csv')
        if not prepared_csv_path or not isinstance(prepared_csv_path, str):
            prepared_csv_path = os.path.join(OUTPUT_DIR, output_cfg['prepared_csv'])

        model_file = os.path.join(MODELS_DIR, output_cfg["model_file"])

        params_file = os.path.join(MODELS_DIR, output_cfg["model_params"])

        best_params = None
        use_auto_tuning = ml_train_cfg.get('use_auto_tuning', False)
            
        if use_auto_tuning:

            if os.path.exists(params_file):
                logger.info("Используем сохраненные гиперпараметры")

                with open(params_file, "r", encoding="utf-8") as f:
                    best_params = json.load(f)

            else:
                logger.info("Подбор гиперпараметров")

                best_params = find_best_catboost_params(
                    input_csv=prepared_csv_path,
                    num_cols_model=ml_predict_cfg["num_cols"],
                    cat_cols=ml_predict_cfg["cat_cols"],
                    target_col="calc_height",
                    random_seed=ml_train_cfg.get("random_seed", 22),
                    cv=ml_train_cfg.get("auto_tuning_cv", 3),
                    n_iter=ml_train_cfg.get("auto_tuning_iterations", 30),
                    verbose=True
                )

                with open(params_file, "w", encoding="utf-8") as f:
                    json.dump(best_params, f, indent=4, ensure_ascii=False)

        logger.info("Обучение финальной модели...")

        model_result = restore_building_height_model(
            input_csv=prepared_csv_path,
            output_model=model_file,
            num_cols_model=ml_predict_cfg["num_cols"],
            cat_cols=ml_predict_cfg["cat_cols"],
            target_col="calc_height",
            random_seed=ml_train_cfg.get("random_seed", 22),
            iterations=ml_train_cfg["iterations"],
            learning_rate=ml_train_cfg["learning_rate"],
            depth=ml_train_cfg["depth"],
            l2_leaf_reg=ml_train_cfg["l2_leaf_reg"],
            best_params=best_params
        )

        state.save(8,{"model_file": model_file, "model_params": params_file})

        current_data["model_file"] = model_file
        current_data["best_params"] = best_params

        logger.info("Шаг 8: выполнен")

        if single_step and start_from == 8:
            return final

        if end_step is not None and 8 >= end_step:
            return final

    else:
        if state.is_step_completed(8):
            model_file = state.get('model_file')
            if model_file:
                current_data['model_file'] = model_file
            logger.info("Шаг 8 пропущен, используем сохраненный")
        else:
            model_file = None
            logger.info("Шаг 8 пропущен")

    # ШАГ 9
    if start_from <= 9 <= end_step:
        logger.info("ШАГ 9: предсказание")
        
        ml_predict_cfg = proc_cfg['ml_predict']
        
        prepared_csv_path = current_data.get('prepared_csv') or state.get('prepared_csv')
        if not prepared_csv_path or not isinstance(prepared_csv_path, str):
            prepared_csv_path = os.path.join(OUTPUT_DIR, output_cfg['prepared_csv'])
        
        predictions_val = predict_with_trained_model(
            input_csv=prepared_csv_path,
            model_path=os.path.join(MODELS_DIR, output_cfg['model_file']),
            output_csv=os.path.join(OUTPUT_DIR, output_cfg['predictions_csv']),
            num_cols=ml_predict_cfg.get('num_cols'),   
            cat_cols=ml_predict_cfg.get('cat_cols')  
        )
        
        predictions_csv = os.path.join(OUTPUT_DIR, output_cfg['predictions_csv'])
        state.save(9, {'predictions_csv': predictions_csv})
        current_data['predictions_csv'] = predictions_csv
        logger.info("Шаг 9: выполнен")
        
        if single_step and start_from == 9:
            return final
        if end_step is not None and 9 >= end_step:
            return final

    else:
        if state.is_step_completed(9):
            predictions_csv = state.get('predictions_csv')
            if predictions_csv:
                current_data['predictions_csv'] = predictions_csv
            logger.info("Шаг 9 пропущен, используем сохраненный")
        else:
            predictions_csv = None
            logger.info("Шаг 9 пропущен")

    # ШАГ 10
    if start_from <= 10 <= end_step:
        logger.info("ШАГ 10: Растеризация высот")
        
        rasterize_cfg = proc_cfg['rasterize_heights']
        predictions_csv = current_data.get('predictions_csv') or state.get('predictions_csv')
        if predictions_csv is None or not isinstance(predictions_csv, str):
            predictions_csv = os.path.join(OUTPUT_DIR, output_cfg['predictions_csv'])
            
        height_calc, height_pred = rasterize_both_heights(
            input_csv=predictions_csv,
            output_dir=OUTPUT_DIR,
            base_name="building_height",
            resolution=RESOLUTION,
            crs_code=CRS_CODE,
            geometry_column=rasterize_cfg.get('geometry_column', 'geometry_wkt'),
            calc_column=rasterize_cfg.get('calc_column', 'calc_height'),
            pred_column=rasterize_cfg.get('pred_column', 'predicted')
        )
        
        state.save(10, {
            'height_calc': height_calc,
            'height_pred': height_pred
        })
        current_data['height_calc'] = height_calc
        current_data['height_pred'] = height_pred
        logger.info("Шаг 10: выполнен")
        
        if single_step and start_from == 10:
            return final
        if end_step is not None and 10 >= end_step:
            return final

    else:
        if state.is_step_completed(10):
            height_calc = state.get('height_calc')
            height_pred = state.get('height_pred')
            if height_calc:
                current_data['height_calc'] = height_calc
            if height_pred:
                current_data['height_pred'] = height_pred
            logger.info("Шаг 10 пропущен, используем сохраненный")
        else:
            height_calc = None
            height_pred = None
            logger.info("Шаг 10 пропущен")

    # ШАГ 11
    if start_from <= 11 <= end_step:
        logger.info("ШАГ 11: Создание новой аллокации")

        buildings_predict = os.path.join(OUTPUT_DIR, output_cfg['height_calc'])

        allocation_correct = calculate_allocation_only(
            input_raster=buildings_predict,
            output_path=os.path.join(OUTPUT_DIR, output_cfg['allocation_correct']),
            resolution=RESOLUTION
        )
        distance_path = os.path.join(OUTPUT_DIR, output_cfg['distance'])
        tiles_path = os.path.join(OUTPUT_DIR, output_cfg['tiles_width'])
        canyon_path = create_empty_multiband_raster(
            output_path=os.path.join(OUTPUT_DIR, output_cfg['canyon_params_final']),
            template_raster=buildings_predict,
            n_bands=proc_cfg['preprocessing'].get('n_output_bands', 3),
            dtype='int16',
            compress='lzw',
            nodata_value=-1
        )

        state.save(11, {
            'allocation_correct': allocation_correct,
            'distance': distance_path,
            'tiles': tiles_path,
            'canyon_params_final': canyon_path
        })

        current_data['allocation_correct'] = allocation_correct
        current_data['distance'] = distance_path
        current_data['tiles'] = tiles_path
        current_data['canyon_params_final'] = canyon_path

        logger.info("Шаг 11: выполнен")

        if single_step and start_from == 11:
            return final
        if end_step is not None and 11 >= end_step:
            return final
        
    else:
        if state.is_step_completed(11):
            allocation_correct = state.get('allocation_correct')
            distance_path = state.get('distance')
            tiles_path = state.get('tiles')
            canyon_path = state.get('canyon_params_final')
            if allocation_correct:
                current_data['allocation_correct'] = allocation_correct
            if distance_path:
                current_data['distance'] = distance_path
            if tiles_path:
                current_data['tiles'] = tiles_path
            if canyon_path:
                current_data['canyon_params_final'] = canyon_path
            logger.info("Шаг 11 пропущен, используем сохраненный")
        else:
            logger.info("Шаг 11 пропущен")

    # ШАГ 12
    if start_from <= 12 <= end_step:
        logger.info("ШАГ 12: ширина каньона")

        wc_cfg = proc_cfg['canyon_width_tiles_second']
        window_size = preproc_cfg['window_size']
        max_radius = wc_cfg.get('max_radius', 500.0)

        width_result = canyon_width_tiles(
            wd=OUTPUT_DIR,
            tiles_file=output_cfg['tiles_width'],
            distance_file=output_cfg['distance'],
            allocation_file=output_cfg['allocation_correct'],
            output_file=output_cfg['canyon_params_final'],  
            skip_existing=False,
            resolution=RESOLUTION,
            window_size=tuple(window_size),
            max_radius=max_radius,
            use_absolute=True
        )
        state.save(12, {
            "canyon_params": width_result["output_file"]
        })

        current_data["canyon_params"] = width_result["output_file"]

        logger.info("Шаг 12: выполнен")

        if single_step and start_from == 12:
            return final
        if end_step is not None and 12 >= end_step:
            return final
        
    else:
        if state.is_step_completed(12):
            canyon_params = state.get('canyon_params')
            if canyon_params:
                current_data['canyon_params'] = canyon_params
            logger.info("Шаг 12 пропущен, используем сохраненный")
        else:
            logger.info("Шаг 12 пропущен")

    # ШАГ 13
    if start_from <= 13 <= end_step:
        logger.info("ШАГ 13: расчет длины каньона")
        
        length_cfg = proc_cfg['canyon_length']
        height = os.path.join(OUTPUT_DIR, output_cfg['height_calc'])
        template_raster = os.path.join(OUTPUT_DIR, output_cfg['buildings_raster'])
        
        canyon_length_result = process_canyon_length(
            tiles_path=os.path.join(OUTPUT_DIR, output_cfg['tiles_width']),
            output_path=os.path.join(OUTPUT_DIR, output_cfg['canyon_length_raster']),
            distance_path=os.path.join(OUTPUT_DIR, output_cfg['distance']),
            allocation_path=height,
            width_path=os.path.join(OUTPUT_DIR, output_cfg['width_raster']),
            target_tiles=None,
            resolution=RESOLUTION,
            max_radius=length_cfg.get('max_radius', 1000.0),
            direction=length_cfg.get('direction', 5.0),
            template_raster=template_raster,
            n_bands=length_cfg.get('n_bands', 4),
            dtype=length_cfg.get('dtype', 'float64'),
            compress=length_cfg.get('compress', 'lzw'),
            nodata_value=length_cfg.get('nodata_value', -1)
        )
        
        state.save(13, {
            'canyon_length': os.path.join(OUTPUT_DIR, output_cfg['canyon_length_raster'])
        })
        current_data['canyon_length'] = os.path.join(OUTPUT_DIR, output_cfg['canyon_length_raster'])
        logger.info("Шаг 13: выполнен")

        if single_step and start_from == 13:
            return final
        if end_step is not None and 13 >= end_step:
            return final

    else:
        if state.is_step_completed(13):
            canyon_length = state.get('canyon_length')
            if canyon_length:
                current_data['canyon_length'] = canyon_length
            logger.info("Шаг 13 пропущен, используем сохраненный")
        else:
            canyon_length = None
            logger.info("Шаг 13 пропущен")

    # ШАГ 14
    if start_from <= 14 <= end_step:
        logger.info("ШАГ 14: Растеризация дорог")

        buildings_raster = current_data.get('buildings_raster') or state.get('buildings_raster')
        
        roads = rasterize_roads(
            roads_path=os.path.join(WD, input_cfg['roads_geojson']),
            building_raster_path=buildings_raster,
            output_path=os.path.join(OUTPUT_DIR, output_cfg['roads_raster']),
            crs_code=CRS_CODE,
            super_sampling=proc_cfg['roads']['super_sampling'],
            use_tqdm=proc_cfg['roads']['use_tqdm']
        )
        
        state.save(14, {'roads': os.path.join(OUTPUT_DIR, output_cfg['roads_raster'])})
        current_data['roads'] = os.path.join(OUTPUT_DIR, output_cfg['roads_raster'])
        logger.info("Шаг 14: выполнен")
        
        if single_step and start_from == 14:
            return final
        if end_step is not None and 14 >= end_step:
            return final

    else:
        if state.is_step_completed(14):
            roads = state.get('roads')
            if roads:
                current_data['roads'] = roads
            logger.info("Шаг 14 пропущен, используем сохраненный")
        else:
            roads = None
            logger.info("Шаг 14 пропущен")

    # ШАГ 15
    if start_from <= 15 <= end_step:
        logger.info("ШАГ 15: Растеризация промышленных зон")
        
        industrial = rasterize_industrial(
            industrial_path=os.path.join(WD, input_cfg['industrial_geojson']),
            building_raster_path=buildings_raster,
            output_path=os.path.join(OUTPUT_DIR, output_cfg['industrial_raster']),
            target_value=proc_cfg['industrial']['target_value'],
            resolution=RESOLUTION,
            crs_code=CRS_CODE
        )
        
        state.save(15, {'industrial': os.path.join(OUTPUT_DIR, output_cfg['industrial_raster'])})
        current_data['industrial'] = os.path.join(OUTPUT_DIR, output_cfg['industrial_raster'])
        logger.info("Шаг 15: выполнен")
        
        if single_step and start_from == 15:
            return final
        if end_step is not None and 15 >= end_step:
            return final

    else:
        if state.is_step_completed(15):
            industrial = state.get('industrial')
            if industrial:
                current_data['industrial'] = industrial
            logger.info("Шаг 15 пропущен, используем сохраненный")
        else:
            industrial = None
            logger.info("Шаг 15 пропущен")

    # ШАГ 16
    if start_from <= 16 <= end_step:
        logger.info("ШАГ 16: Перепроецирование World Cover")
        
        worldcover = reproject_resample_clip(
            input_rasters=os.path.join(WD, input_cfg['worldcover_pattern']),
            reference_raster=buildings_raster,
            output_path=os.path.join(OUTPUT_DIR, output_cfg['worldcover']),
            crs_code=CRS_CODE,
            resolution=RESOLUTION,
            resampling=proc_cfg['worldcover']['resampling'],
            crop_before=proc_cfg['worldcover'].get('crop_before', False),       
            buffer_meters=proc_cfg['worldcover'].get('buffer_meters', 0) 
        )
        
        state.save(16, {'worldcover': os.path.join(OUTPUT_DIR, output_cfg['worldcover'])})
        current_data['worldcover'] = os.path.join(OUTPUT_DIR, output_cfg['worldcover'])
        logger.info("Шаг 16: выполнен")
        
        if single_step and start_from == 16:
            return final
        if end_step is not None and 16 >= end_step:
            return final

    else:
        if state.is_step_completed(16):
            worldcover = state.get('worldcover')
            if worldcover:
                current_data['worldcover'] = worldcover
            logger.info("Шаг 16 пропущен, используем сохраненный")
        else:
            worldcover = None
            logger.info("Шаг 16 пропущен")

    # ШАГ 17
    if start_from <= 17 <= end_step:
        logger.info("ШАГ 17: Перепроецирование LCZ")
        
        lcz = reproject_resample_clip(
            input_rasters=os.path.join(WD, input_cfg['lcz_raster']),
            reference_raster=buildings_raster,
            output_path=os.path.join(OUTPUT_DIR, output_cfg['lcz_correct']),
            crs_code=CRS_CODE,
            resolution=RESOLUTION,
            band_number=proc_cfg['lcz_reproject'].get('band_number', 1),
            resampling=proc_cfg['lcz_reproject']['resampling'],
            crop_before=proc_cfg['lcz_reproject'].get('crop_before', False),     
            buffer_meters=proc_cfg['lcz_reproject'].get('buffer_meters', 0)
        )
        
        state.save(17, {'lcz': os.path.join(OUTPUT_DIR, output_cfg['lcz_correct'])})
        current_data['lcz'] = os.path.join(OUTPUT_DIR, output_cfg['lcz_correct'])
        logger.info("Шаг 17: выполнен")
        
        if single_step and start_from == 17:
            return final
        if end_step is not None and 17 >= end_step:
            return final

    else:
        if state.is_step_completed(17):
            lcz = state.get('lcz')
            if lcz:
                current_data['lcz'] = lcz
            logger.info("Шаг 17 пропущен, используем сохраненный")
        else:
            lcz = None
            logger.info("Шаг 17 пропущен")

    # ШАГ 18
    if start_from <= 18 <= end_step:
        logger.info("ШАГ 18: Перепроецирование FABDEM")
        
        fabdem_tiles_pattern = input_cfg['fabdem_tiles_pattern']
        fabdem_tiles = glob.glob(os.path.join(WD, fabdem_tiles_pattern))
        
        fabdem = reproject_resample_clip(
            input_rasters=fabdem_tiles,
            reference_raster=buildings_raster,
            output_path=os.path.join(OUTPUT_DIR, output_cfg['fabdem_resampled']),
            crs_code=CRS_CODE,
            resolution=RESOLUTION,
            resampling=proc_cfg['reproject']['resampling'],
            crop_before=proc_cfg['reproject'].get('crop_before', False),         
            buffer_meters=proc_cfg['reproject'].get('buffer_meters', 0)   
        )
        
        state.save(18, {'fabdem': os.path.join(OUTPUT_DIR, output_cfg['fabdem_resampled'])})
        current_data['fabdem'] = os.path.join(OUTPUT_DIR, output_cfg['fabdem_resampled'])
        logger.info("Шаг 18: выполнен")
        
        if single_step and start_from == 18:
            return final
        if end_step is not None and 18 >= end_step:
            return final

    else:
        if state.is_step_completed(18):
            fabdem = state.get('fabdem')
            if fabdem:
                current_data['fabdem'] = fabdem
            logger.info("Шаг 18 пропущен, используем сохраненный")
        else:
            fabdem = None
            logger.info("Шаг 18 пропущен")

    # ШАГ 19
    if start_from <= 19 <= end_step:
        logger.info("ШАГ 19: Создание финального многоканального растра")
        
        multiband_cfg = config['processing']['multiband']
        final_output_path = os.path.join(READY_DIR, output_cfg['final_raster'])
        
        target_crs = config.get('target_crs')
        
        discrete_bands = multiband_cfg.get('discrete_bands', None)
        
        band_files = [os.path.join(OUTPUT_DIR, b['source']) for b in multiband_cfg['bands']]
        band_names = [b['name'] for b in multiband_cfg['bands']]
        band_indices = [b['band'] for b in multiband_cfg['bands']]
        
        final = create_multiband_raster(
            output_path=final_output_path,
            template_raster=buildings_raster,
            band_files=band_files,
            band_names=band_names,
            band_indices=band_indices,
            nodata_value=multiband_cfg['nodata_value'],
            dtype=multiband_cfg['dtype'],
            compress=multiband_cfg['compress'],
            target_crs=target_crs,
            discrete_bands=discrete_bands
        )
        
        state.save(19, {'final': final_output_path})
        current_data['final'] = final_output_path
        logger.info("Шаг 19: выполнен")
        
        if single_step and start_from == 19:
            return final
        if end_step is not None and 19 >= end_step:
            return final

    else:
        if state.is_step_completed(19):
            final = state.get('final')
            if final:
                current_data['final'] = final
            logger.info("Шаг 19 пропущен, используем сохраненный")
        else:
            final = None
            logger.info("Шаг 19 пропущен")

    logger.info("ОБРАБОТКА ЗАВЕРШЕНА")

    return final

# ВЫПОЛНЕНИЕ ОДНОГО ШАГА 

def run_single_step(config_path: str, step: int):
    print(f"Запуск только шага {step}")
    run_pipeline(
        config_path=config_path, 
        start_from=step, 
        single_step=True
    )

# ЧТЕНИЕ ФАЙЛА СОСТОЯНИЯ 
def show_pipeline_status(config_path: str = "config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    WD = config['wd']
    OUTPUT_DIR = os.path.join(WD, config['output_dir'])
    state_file = os.path.join(OUTPUT_DIR, 'pipeline_state.pkl')
    
    if not os.path.exists(state_file):
        print("\nФайл состояния не найден")
        return None
    
    with open(state_file, 'rb') as f:
        data = pickle.load(f)
    
    current_step = data['current_step']
    completed_steps = set(data.get('completed_steps', []))
    state = data['state']
    timestamp = data.get('timestamp', 'неизвестно')
 
    steps_info = {
        1: "Растеризация зданий",
        2: "Препроцессинг",
        3: "Расчет ширины каньона (предварительный)",
        4: "Фильтрация ширины",
        5: "Обработка DEM",
        6: "Обработка LCZ",
        7: "Подготовка данных для модели",
        8: "Обучение модели",
        9: "Предсказание по модели",
        10: "Растеризация высот",
        11: "Создание новой аллокации",
        12: "Расчет ширины каньона (финальный)",
        13: "Расчет длины каньона",
        14: "Растеризация дорог",
        15: "Растеризация промышленных зон",
        16: "Перепроецирование World Cover",
        17: "Перепроецирование LCZ",
        18: "Перепроецирование FABDEM",
        19: "Финальный многоканальный растр",
    }

    def get_step_time(step, state_dict):
        time_key = f'_step_{step}_time'
        return state_dict.get(time_key)
    
    print(f"\nПоследнее обновление: {timestamp}")
    print(f"Текущий шаг: {current_step}")
    print(f"Выполненные шаги: {sorted(completed_steps)}")
    
    for step in range(1, 20):
        if step in completed_steps:
            step_time = get_step_time(step, state)
            if step_time:
                time_str = step_time.split('T')[1][:8] if 'T' in step_time else step_time
                print(f"   Шаг {step:2d}: {steps_info[step]} [{time_str}]")
            else:
                print(f"   Шаг {step:2d}: {steps_info[step]}")

    return {
        'current_step': current_step,
        'completed_steps': sorted(completed_steps),
        'timestamp': timestamp,
        'is_complete': len(completed_steps) >= 19
    }

# ИСПОЛЬЗОВАНИЕ

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск расчета параметров")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--start-from", "-s", type=int, default=None)
    parser.add_argument("--only-step", "-o", type=int, default=None)
    parser.add_argument("--range", "-rng", type=str, default=None)
    parser.add_argument("--steps", "-stp", type=str, default=None)
    parser.add_argument("--reset", "-r", action="store_true")
    parser.add_argument("--status", "-st", action="store_true")
    
    args = parser.parse_args()
    
    if args.status:
        show_pipeline_status(args.config)

    elif args.steps is not None:
        steps_list = [int(s.strip()) for s in args.steps.split(',')]
        print(f"Запуск выбранных шагов: {steps_list}")
        
        for step in steps_list:
            print(f"\n--- Выполнение шага {step} ---")
            run_pipeline(
                config_path=args.config,
                start_from=step,
                end_step=step,
                single_step=True
            )

    elif args.only_step is not None:
        run_pipeline(
            config_path=args.config, 
            start_from=args.only_step, 
            single_step=True
        )

    elif args.range is not None:
        start_step, end_step = map(int, args.range.split('-'))
        print(f"Запуск шагов с {start_step} по {end_step}")
        run_pipeline(
            config_path=args.config,
            start_from=start_step,
            end_step=end_step,
            single_step=False
        )
        
    else:
        if args.reset:
            config = load_config(args.config)
            WD = config['wd']
            OUTPUT_DIR = os.path.join(WD, config['output_dir'])
            state_file = os.path.join(OUTPUT_DIR, 'pipeline_state.pkl')
            if os.path.exists(state_file):
                os.remove(state_file)
                print(f"Состояние сброшено: {state_file}")
        
        run_pipeline(
            config_path=args.config, 
            start_from=args.start_from, 
            single_step=False
        )