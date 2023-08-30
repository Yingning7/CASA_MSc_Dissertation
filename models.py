from typing import Optional, Union, Any, Callable
from copy import deepcopy

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def pack_results(
    dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]],
    model: Any
) -> dict[Any]:
    dataset = deepcopy(dataset)
    results = {
        'dataset': dataset,
        'model': model,
        'predict': {
            'train': model.predict(dataset['train']['X']),
            'test': model.predict(dataset['test']['X'])
        },
        'score': {
            'train': {
                'rmse': None,
                'r2': None
            },
            'test': {
                'rmse': None,
                'r2': None
            }
        }
    }
    for name in ['train', 'test']:
        results['score'][name]['rmse'] = mean_squared_error(
            dataset[name]['y'],
            model.predict(dataset[name]['X']),
            squared=False
        )
        results['score'][name]['r2'] = r2_score(
            dataset[name]['y'],
            model.predict(dataset[name]['X'])
        )
    return results


def wrap_results(model_func: Callable) -> Callable:
    def inner(dataset: dict, *args, **kwargs) -> dict[Any]:
        model = model_func(dataset, *args, **kwargs)
        results = pack_results(dataset, model)
        return results
    return inner


@wrap_results
def lr(dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]]) -> Any:
    model = LinearRegression(fit_intercept=False).fit(
        dataset['train']['X'],
        dataset['train']['y']
    )
    return model


@wrap_results
def ridge(
    dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]], 
    alpha: float
) -> Any:
    model = Ridge(fit_intercept=False, alpha=alpha).fit(
        dataset['train']['X'],
        dataset['train']['y']
    )
    return model


@wrap_results
def lasso(
    dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]], 
    alpha: float
) -> Any:
    model = Lasso(fit_intercept=False, alpha=alpha).fit(
        dataset['train']['X'],
        dataset['train']['y']
    )
    return model


@wrap_results
def tree(
    dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]],
    max_depth: Optional[int] = None
) -> Any:
    model = DecisionTreeRegressor(max_depth=max_depth).fit(
        dataset['train']['X'],
        dataset['train']['y']
    )
    return model


@wrap_results
def xgb(
    dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]],
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None
) -> Any:
    model = XGBRegressor(max_depth=max_depth).fit(
        dataset['train']['X'],
        dataset['train']['y']
    )
    return model


@wrap_results
def nn(
    dataset: dict[str, dict[str, Union[pd.DataFrame, pd.Series]]],
    hidden_layer_sizes: tuple[int, ...] = (100,),
    alpha: float = 0.0001,
    activation: str = 'relu'
) -> Any:
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        activation=activation,
        max_iter=5000,
        learning_rate_init=0.01
    ).fit(
        dataset['train']['X'],
        dataset['train']['y']
    )
    return model
