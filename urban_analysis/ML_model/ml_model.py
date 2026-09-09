"""
Модель восстановления высот зданий
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool


def find_best_catboost_params(
    input_csv: str,
    num_cols_model: list = None,
    cat_cols: list = None,
    target_col: str = "calc_height",
    random_seed: int = 22,
    cv: int = 3,
    n_iter: int = 30,
    verbose: bool = True
) -> dict:
    """
    Автоматический подбор гиперпараметров CatBoost
    с использованием randomized_search().

    Возвращает словарь лучших параметров.
    """
    dataset = pd.read_csv(input_csv)

    if num_cols_model is None:
        num_cols_model = [
            "area",
            "isoquotient",
            "obox_ratio",
            "width_med",
            "obox_hw",
            "min_dem",
            "neighbors_100m"  
        ]

    if cat_cols is None:
        cat_cols = [
            "majority_100m",
            "class",
            "subtype"
        ]

    feature_cols = num_cols_model + cat_cols

    clean = dataset[feature_cols + [target_col]].dropna()
    clean = clean[clean[target_col] >= 1]
    clean = clean[clean[target_col] > 0]

    X = clean[feature_cols].copy()
    y = clean[target_col].copy()

    for col in cat_cols:
        X[col] = X[col].astype(str)

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    train_pool = Pool(
        data=X,
        label=y,
        cat_features=cat_indices
    )

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_seed,
        verbose=False
    )

    param_distributions = {
        "depth": [4, 5, 6, 7, 8],
        "iterations": [800, 1000, 1500, 2000, 2500],
        "learning_rate": [0.01, 0.015, 0.02, 0.03, 0.05],
        "l2_leaf_reg": [1, 3, 5, 7, 10, 15],
        "bagging_temperature": [0, 1, 3, 5],
        "random_strength": [0.5, 1, 2, 5]
    }

    result = model.randomized_search(
        param_distributions=param_distributions,
        X=train_pool,
        cv=cv,
        n_iter=n_iter,
        partition_random_seed=random_seed,
        shuffle=True,
        calc_cv_statistics=True,
        verbose=verbose,
        plot=False
    )

    best_params = result["params"]

    return best_params


def restore_building_height_model(
    input_csv: str,
    output_model: str,
    num_cols_model: list = None,
    cat_cols: list = None,
    target_col: str = "calc_height",
    random_seed: int = 22,
    iterations: int = 1000,
    learning_rate: float = 0.03,
    depth: int = 4,
    l2_leaf_reg: int = 10,
    bagging_temperature: float = 1,
    random_strength: float = 1,
    best_params: dict = None    
) -> dict:
    """
    Обучение модели CatBoost для предсказания высоты зданий
    
    Args:
        input_csv: путь к CSV файлу с данными
        output_model: путь для сохранения модели
        num_cols_model: список числовых колонок для обучения
        cat_cols: список категориальных колонок для обучения
        target_col: название целевой колонки
        random_seed: случайное зерно
        iterations: количество итераций
        learning_rate: скорость обучения
        depth: глубина деревьев
        l2_leaf_reg: L2 регуляризация
        best_params: словарь с лучшими параметрами из find_best_catboost_params
    
    Returns:
        dict: словарь с результатами
    """
    dataset = pd.read_csv(input_csv)
    print(f"  Загружено записей: {len(dataset)}")
    
    if num_cols_model is None:
        num_cols_model = [
            'area',
            'isoquotient',
            'obox_ratio',
            'width_med',
            'obox_hw',
            'min_dem',
            'neighbors_100m'  
        ]
    
    if cat_cols is None:
        cat_cols = ['majority_100m', 'class', 'subtype']
    
    all_cols = num_cols_model + cat_cols + [target_col]
    missing_cols = [col for col in all_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Следующие колонки отсутствуют в данных: {missing_cols}")

    clean = dataset[all_cols].dropna()
    clean = clean[clean[target_col] >= 1.0]
    clean = clean[clean[target_col] > 0]
    print(f"  После удаления пропусков: {len(clean)} записей")
    
    if len(clean) == 0:
        raise ValueError("Нет данных для обучения после удаления пропусков!")

    if best_params is not None:
        iterations = best_params.get("iterations", iterations)
        learning_rate = best_params.get("learning_rate", learning_rate)
        depth = best_params.get("depth", depth)
        l2_leaf_reg = best_params.get("l2_leaf_reg", l2_leaf_reg)
        bagging_temperature = best_params.get("bagging_temperature", bagging_temperature)
        random_strength = best_params.get("random_strength", random_strength)

    feature_cols = num_cols_model + cat_cols
    X = clean[feature_cols].copy()
    Y = clean[target_col].copy()

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    model = CatBoostRegressor(
        cat_features=cat_cols,  
        loss_function="RMSE",
        random_seed=random_seed,
        verbose=100,  
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        random_strength=random_strength,
        early_stopping_rounds=50,
        eval_metric='RMSE'
    )

    model.fit(X, Y)
    model.save_model(output_model)

    return {
        'model': model,
        'model_path': output_model,
        'n_samples': len(clean),
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'best_params': best_params
    }


def predict_with_trained_model(
    input_csv: str,
    model_path: str,
    output_csv: str,
    num_cols: list = None,
    cat_cols: list = None
) -> pd.DataFrame:
    """
    Предсказание высот зданий с использованием обученной модели
    Заполняет пропуски в calc_height предсказанными значениями
    Добавляет колонку predicted: 1 - предсказано, 0 - исходное значение
    """
    dataset = pd.read_csv(input_csv)
    
    if num_cols is None:
        num_cols = [
            'area',
            'isoquotient',
            'obox_ratio',
            'width_med',
            'obox_hw',
            'min_dem',
            'neighbors_100m'  
        ]
    
    if cat_cols is None:
        cat_cols = ['majority_100m', 'class', 'subtype']
    
    feature_cols = num_cols + cat_cols

    final_model = CatBoostRegressor()
    final_model.load_model(model_path)

    X_full = dataset[feature_cols].copy()
    

    for col in cat_cols:
        X_full[col] = X_full[col].fillna('unknown')
        X_full[col] = X_full[col].astype(str)
    
    for col in num_cols:
        X_full[col] = pd.to_numeric(X_full[col], errors='coerce')
    
    X_full[num_cols] = X_full[num_cols].fillna(0)

    preds = final_model.predict(X_full)
    preds_rounded = np.maximum(0, np.round(preds)).astype(int)

    mask_missing = (dataset['calc_height'].isna()) | (dataset['calc_height'] == 0)
    dataset.loc[mask_missing, 'calc_height'] = preds_rounded[mask_missing]
    dataset['predicted'] = mask_missing.astype(int)
    
    dataset.to_csv(output_csv, index=False, encoding='utf-8')
    
    return dataset