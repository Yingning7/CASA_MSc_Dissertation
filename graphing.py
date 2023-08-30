from typing import Any
from copy import deepcopy
import pickle
import json

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from data_utils import load_reses


def best_xgb_mapping(t_as: int, year: int, **kwargs) -> dict[str, go.Figure]:
    def _mapping(df: pd.DataFrame, gj: dict[Any], t_as: int, title: str, **kwargs) -> go.Figure:
        fig = px.choropleth_mapbox(
            data_frame=df,
            geojson=gj,
            featureidkey=kwargs.pop('featureidkey', 'properties.code'),
            locations=kwargs.pop('locations', 'code'),
            color=kwargs.pop('color', 'population_{{t+{t_as}}}').format(t_as=t_as),
            mapbox_style=kwargs.pop('mapbox_style', 'open-street-map'),
            center=kwargs.pop('center', {'lat': 53, 'lon': -1.5}),
            zoom=kwargs.pop('zoom', 6),
            range_color=kwargs.pop('range_color', [30000, 1250000]),
            height=kwargs.pop('height', 950),
            width=kwargs.pop('width', 750),
            title=title,
            **kwargs
        ).update_layout(margin={'r': 0, 't': 80, 'l': 0, 'b': 0})
        return fig
    
    with open('C:/Users/danie/Desktop/yingning/geojson/combined.geojson', 'r') as fp:
        gj = json.load(fp)
    with open(f'C:/Users/danie/Desktop/yingning/saves/xgb_n_estimators5_max_depth3/(False, {t_as}, False).pickle', 'rb') as fp:
        res = pickle.load(fp)
    full_X = pd.concat([res['dataset']['train']['X'], res['dataset']['test']['X']], axis=0)
    full_y = pd.concat([res['dataset']['train']['y'], res['dataset']['test']['y']], axis=0)
    pred_y = pd.Series(res['model'].predict(full_X), index=full_y.index).rename(full_y.name)
    full_y_df = full_y.reset_index()
    full_y_df = full_y_df.loc[full_y_df['year'] == year].drop('year', axis=1)
    pred_y_df = pred_y.reset_index()
    pred_y_df = pred_y_df.loc[pred_y_df['year'] == year].drop('year', axis=1)
    maps = {
        'actual': _mapping(full_y_df, gj, t_as, f'Map View: Actual Population in {year + t_as}', **deepcopy(kwargs)),
        'predict': _mapping(pred_y_df, gj, t_as, f'Map View: XGB Regression Predicted Population in {year + t_as} (as t+{t_as})<br>with features in {year - 1} (t-1) and {year - 2} (t-2)', **deepcopy(kwargs))
    }
    return maps


def lr_driver_analysis(
    t_as: int, 
    year_1: int, 
    year_2: int, 
    splits: tuple[int, int] = (100e3, 420e3)
) -> dict[str, go.Figure]:
    def _driver_analysis_graph(driver: pd.DataFrame, year_1: int, year_2: int, title_suffix: str = ''):
        year_mean = driver.groupby(level='year').mean()
        diff = year_mean.loc[year_2] - year_mean.loc[year_1]
        diff_pct = diff / year_mean.loc[year_1].abs()
        diff = diff.to_frame().T.melt(value_name='mean_diff')
        diff_pct = diff_pct.to_frame().T.melt(value_name='mean_diff_pct')
        da = {
            'diff': px.bar(
                diff,
                x='variable',
                y='mean_diff',
                barmode='group',
                title=f'Driver Analysis on changes between {year_1} and {year_2}{title_suffix}',
                template='plotly_white'
            ).update_layout(
                xaxis_title='Drivers',
                yaxis_title='Mean Difference of Driver Values',
                legend_title=None
            ),
            'diff_pct': px.bar(
                diff_pct,
                x='variable',
                y='mean_diff_pct',
                barmode='group',
                title=f'Driver Analysis on percentage changes between {year_1} and {year_2}{title_suffix}',
                template='plotly_white'
            ).update_layout(
                xaxis_title='Drivers',
                yaxis_title='Mean Percentage Difference of Driver Values',
                legend_title=None
            ),
        }
        return da
    
    key = f'(True, {t_as}, False)'
    res = load_reses('lr')
    full_X = pd.concat([res[key]['dataset']['train']['X'], res[key]['dataset']['test']['X']], axis=0)
    full_y = pd.concat([res[key]['dataset']['train']['y'], res[key]['dataset']['test']['y']])
    coef = pd.Series(res[key]['model'].coef_, index=full_X.columns, name='coef')
    driver = coef * full_X
    da = {}
    da['full'] = _driver_analysis_graph(driver, year_1, year_2, title_suffix=' [full]')
    da['low'] = _driver_analysis_graph(driver.loc[full_y < splits[0]], year_1, year_2, title_suffix=' [low]')
    da['medium'] = _driver_analysis_graph(driver.loc[(splits[0] < full_y) & (full_y <= splits[1])], year_1, year_2, title_suffix=' [medium]')
    da['high'] = _driver_analysis_graph(driver.loc[splits[1] < full_y], year_1, year_2, title_suffix=' [high]')
    return da
