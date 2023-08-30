from typing import Optional, Union, Any
from itertools import product
from copy import deepcopy
from pathlib import Path
import pickle
import json
import os

from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.model_selection import train_test_split
import pandas as pd


def fetch_usable_data(
    keep_raw_features_cols: Optional[list[str, ...]] = None,
    sum_cols: Optional[list[str, ...]] = None,
    add_intercept: bool = True
) -> pd.DataFrame:
    def sum_cols_to_total(data: pd.DataFrame, key: str) -> pd.DataFrame:
        df = data.copy()
        cols = df.columns[df.columns.str.contains(f'{key}_')]
        key_sum = df[cols].sum(axis=1)
        df = df.drop(cols, axis=1)
        df[f'{key}_total'] = key_sum
        return df
    
    data = pd.read_csv('C:/Users/danie/Desktop/yingning/data/final.csv')
    data = data.drop('gva_total', axis=1)
    data = data[data.columns[~data.columns.str.contains('name_')]]
    if keep_raw_features_cols is not None:
        data = data[['year', 'code', 'population'] + keep_raw_features_cols]
    if sum_cols is not None:
        if 'crime' in sum_cols:
            data = sum_cols_to_total(data, 'crime')
        if 'gva' in sum_cols:
            data = sum_cols_to_total(data, 'gva')
        if 'ghg' in sum_cols:
            data = sum_cols_to_total(data, 'ghg')
    data = data.dropna()
    if add_intercept:
        data['const'] = 1.
    data = data[sorted(data.columns)]
    data = data.sort_values(['year', 'code']).set_index(['year', 'code'])
    return data


def time_shift(
    data: pd.DataFrame,
    lags: Optional[list[int, ...]] = None,
    forwards: Optional[list[int, ...]] = None,
    forward_col: str = 'population'
) -> pd.DataFrame:
    df = data.copy().reset_index().sort_values(['year', 'code']).reset_index(drop=True)
    has_const = 'const' in df.columns
    if has_const:
        df = df.drop('const', axis=1)
    concat_dfs = []
    if lags is not None:
        for i in lags:
            lagged = df.drop('year', axis=1).groupby('code').shift(i)
            lagged.columns += f'_{{t-{i}}}'
            concat_dfs.append(lagged)
    if forwards is not None:
        for i in forwards:
            forwarded = df[['code', forward_col]].groupby('code').shift(-i)
            forwarded.columns += f'_{{t+{i}}}'
            concat_dfs.append(forwarded)
    shifted = pd.concat([df[['year', 'code']]] + concat_dfs, axis=1)
    if 0 in lags and 0 in forwards:  # avoid t+0 and t-0 on the same col
        shifted = shifted.drop(f'{forward_col}_{{t-0}}', axis=1)
    shifted = shifted.dropna()
    if has_const:
        shifted['const'] = 1.
    shifted = shifted[sorted(shifted.columns)]
    shifted = shifted.set_index(['year', 'code'])
    return shifted


def split(data: pd.DataFrame, y_col: str, train_end_year: int = 2015, standardise: bool = True) -> dict[str, dict[Union[pd.DataFrame, pd.Series]]]:
    data_temp = data.copy().reset_index()
    train = data_temp.loc[data_temp['year'] <= train_end_year].copy().set_index(['year', 'code'])
    test = data_temp.loc[data_temp['year'] > train_end_year].copy().set_index(['year', 'code'])
    splits = (
        train.drop(y_col, axis=1).copy(),
        test.drop(y_col, axis=1).copy(),
        train[y_col].copy(),
        test[y_col].copy()
    )
    dataset = {
        'train': {
            'X': splits[0],
            'y': splits[2]
        },
        'test': {
            'X': splits[1],
            'y': splits[3]
        }
    }
    if standardise:
        has_const = 'const' in dataset['train']['X']
        if has_const:
            dataset['train']['X'] = dataset['train']['X'].drop('const', axis=1)
            dataset['test']['X'] = dataset['test']['X'].drop('const', axis=1)
        mean = dataset['train']['X'].mean(axis=0)
        std = dataset['train']['X'].std(axis=0)
        dataset['train']['X'] = (dataset['train']['X'] - mean) / std
        dataset['test']['X'] = (dataset['test']['X'] - mean) / std
        if has_const:
            dataset['train']['X']['const'] = 1.
            dataset['test']['X']['const'] = 1.
        dataset['train']['X'] = dataset['train']['X'][sorted(dataset['train']['X'].columns)]
        dataset['test']['X'] = dataset['test']['X'][sorted(dataset['test']['X'].columns)]
    return dataset


def fetch_selected_dataset(
    add_intercept: bool,
    single_forward: int,
    standardise: bool
) -> dict[str, dict[Union[pd.DataFrame, pd.Series]]]:
    keep_raw_features_cols = [
        'crime_All other theft offences', 'crime_Bicycle theft',
        'crime_Criminal damage and arson',
        'crime_Death or serious injury caused by illegal driving',
        'crime_Domestic burglary', 'crime_Drug offences',
        'crime_Fraud offences', 'crime_Homicide',
        'crime_Miscellaneous crimes against society',
        'crime_Non-domestic burglary', 'crime_Non-residential burglary',
        'crime_Possession of weapons offences', 'crime_Public order offences',
        'crime_Residential burglary', 'crime_Robbery', 'crime_Sexual offences',
        'crime_Shoplifting', 'crime_Stalking and harassment',
        'crime_Theft from the person', 'crime_Vehicle offences',
        'crime_Violence with injury', 'crime_Violence without injury',

        'ghg_Agriculture', 'ghg_Commercial',
        'ghg_Domestic', 'ghg_Industry', 'ghg_LULUCF', 'ghg_Public Sector',
        'ghg_Transport', 'ghg_Waste management',

        'gcse',
        'gva_ABDE',
        'gva_C',
        'gva_F',
        'gva_G',
        'gva_H',
        'gva_K',
        'gva_N',
        'gva_Q',
        'road-length',
        'tax',
        'unemployment'
    ]
    data = fetch_usable_data(
        keep_raw_features_cols=keep_raw_features_cols, 
        sum_cols=['crime', 'ghg'], 
        add_intercept=add_intercept
    )
    shifted = time_shift(
        data, 
        lags=[1, 2], 
        forwards=[single_forward]
    ).drop(['population_{t-1}', 'population_{t-2}'], axis=1)
    dataset = split(
        shifted, 
        f'population_{{t+{single_forward}}}', 
        standardise=standardise
    )
    return dataset


def fetch_geojson() -> dict[Any]:
    with open('C:/Users/danie/Desktop/yingning/geojson/combined.geojson', mode='r') as fp:
        geojson = json.load(fp)
    return geojson


def save_reses(reses: dict[Any], folder_name: str) -> None:
    path = Path(f'C:/Users/danie/Desktop/yingning/saves/{folder_name}')
    path.mkdir(parents=True, exist_ok=True)
    for name, res in reses.items():
        with open(path / Path(f'{name}.pickle'), mode='wb') as fp:
            pickle.dump(res, fp)


def load_reses(folder_name: str) -> dict[Any]:
    directory = Path(f'C:/Users/danie/Desktop/yingning/saves/{folder_name}')
    file_names = os.listdir(directory)
    reses = {}
    for file_name in file_names:
        with open(directory / Path(file_name), mode='rb') as fp:
            reses[file_name.split('.pickle')[0]] = pickle.load(fp)
    return reses


def extract_reses_metrics(reses: dict[Any]) -> pd.DataFrame:
    dfs = []
    for key, res in reses.items():
        with_intercept, pred_t, with_standardisation = [token.strip() for token in key[1:-1].split(',')]
        with_intercept = with_intercept == 'True'
        pred_t = int(pred_t)
        with_standardisation = with_standardisation == 'True'
        for typ in ['train', 'test']:
            dfs.append(
                pd.Series(
                    [
                        res['score'][typ]['r2'],
                        res['score'][typ]['rmse'], 
                        with_intercept,
                        with_standardisation,
                        pred_t, 
                        typ
                    ], 
                    index=[
                        'r2',
                        'rmse', 
                        'intercept',
                        'standardisation',
                        'predict_t+', 
                        'type']
                ).to_frame().T
            )
    table = pd.concat(dfs, axis=0, ignore_index=True).infer_objects().sort_values(
        ['type', 'predict_t+', 'intercept', 'standardisation']
    ).reset_index(drop=True)
    table['predict_t+'] = table['predict_t+'].astype(str)
    return table


def construct_metrics_table(model: str) -> list[tuple[Any]]:
    match model:
        case 'lr':
            hyper = {}
            folder_name = 'lr'
        case 'lasso':
            hyper = {
                'alpha': [0.001, 0.01, 0.1, 1., 10., 100., 1000.]
            }
            folder_name = 'lasso_alpha{0}'
        case 'ridge':
            hyper = {
                'alpha': [0.001, 0.01, 0.1, 1., 10., 100., 1000.]
            }
            folder_name = 'ridge_alpha{0}'
        case 'tree': 
            hyper = {
                'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
            }
            folder_name = 'tree_max_depth{0}'
        case 'xgb':
            hyper = {
                'n_estimators': [5, 10, 20, 40, 60, 100, 150, 200],
                'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
            }
            folder_name = 'xgb_n_estimators{0}_max_depth{1}'
        case 'nn':
            hyper = {
                'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 128)],
                'alpha': [0.001, 0.01, 0.1, 1.],
                'activation': ['relu', 'tanh']
            }
            folder_name = 'nn_hidden_layer_sizes{0}_alpha{1}_activation{2}'
    if len(hyper) == 0:
        reses = load_reses(folder_name)
        table = extract_reses_metrics(reses)
    else:
        dfs = []
        hyper_iter = list(product(*[v for k, v in hyper.items()]))
        for hyper_tuple in hyper_iter:
            reses = load_reses(folder_name.format(*hyper_tuple))
            df = extract_reses_metrics(reses)
            for k, v in zip(hyper.keys(), hyper_tuple):
                if isinstance(v, tuple):
                    df[k] = [v] * len(df.index)
                else:
                    df[k] = v
            dfs.append(df)
        table = pd.concat(dfs, axis=0, ignore_index=True)
    return table


def filter_metrics(metrics: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    df = metrics.copy()
    for k, v in filters.items():
        df = df.loc[df[k] == v]
    df = df.reset_index(drop=True)
    return df


def fetch_best_model_metrics():
    dfs = []
    for model in ['lr', 'lasso', 'ridge', 'tree', 'xgb', 'nn']:
        metrics = construct_metrics_table(model)
        match model:
            case 'lr': 
                filtered_metrics = filter_metrics(metrics, intercept=True, standardisation=False)
            case 'lasso':
                filtered_metrics = filter_metrics(metrics, intercept=True, standardisation=True, alpha=1000.)
            case 'ridge':
                filtered_metrics = filter_metrics(metrics, intercept=True, standardisation=True, alpha=10.)
            case 'tree':
                filtered_metrics = filter_metrics(metrics, intercept=False, standardisation=False, max_depth=12)
            case 'xgb':
                filtered_metrics = filter_metrics(metrics, intercept=False, standardisation=False, max_depth=3, n_estimators=5)
            case 'nn':
                filtered_metrics = filter_metrics(metrics, intercept=True, standardisation=True, activation='relu', hidden_layer_sizes=(128,), alpha=1.)
        filtered_metrics['model'] = model
        dfs.append(filtered_metrics)
    best_metrics = pd.concat(dfs, axis=0, ignore_index=True)
    return best_metrics
