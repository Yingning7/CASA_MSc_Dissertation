from typing import Union
from copy import deepcopy

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def _vif(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = data.copy()
    while True:
        vif_df = pd.Series(
            [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
            name= "VIF",
            index=df.columns
        ).to_frame().drop('const')
        if vif_df['VIF'].max() > threshold:
            idx_drop = vif_df.index[vif_df.VIF == vif_df.VIF.max()].tolist()[0]
            df = df.drop(columns=idx_drop)
        else:
            break
    return df


def vif(dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]], threshold: float) -> dict[str, dict[str, Union[pd.DataFrame, pd.Series]]]:
    dataset = deepcopy(dataset)
    cols = _vif(dataset['train']['X'], threshold).columns
    dataset['train']['X'] = dataset['train']['X'][cols]
    dataset['test']['X'] = dataset['test']['X'][cols]
    return dataset
