# CASA MSc Dissertation: Spatiotemporal Population Prediction

This repository contains the codebase for a Master's dissertation at the Centre for Advanced Spatial Analysis (CASA). The project aims to predict future population levels at small geographic scales by utilizing various socioeconomic, environmental, and temporal features.

## Project Overview

The core objective is to forecast population dynamics across different regions (`population_{t+n}`) through historical predictors (`t-1`, `t-2`). The approach implements and compares multiple regression machine learning models to identify the key thematic drivers (like crime, greenhouse gas emissions, and economic activities) that correspond to local population growth or decline.

## Repository Structure

### Core Modules
- **`data_utils.py`**: Contains data loading, processing, and feature engineering logic. This handles calculating lagged time-series variations, partitioning train/test sets, feature scaling, and standardizing inputs.
- **`dimensionality_reduction.py`**: Selects non-collinear features using iterative Variance Inflation Factor (VIF) procedures.
- **`models.py`**: Defines and evaluates predictive regression models—Linear Regression, Ridge, Lasso, Decision Tree, XGBoost, and Multilayer Perceptron (Neural Network)—and yields robust test performance scores (R² and RMSE).
- **`graphing.py`**: Generates geospatial visualizations using `plotly` and `mapbox`, showcasing side-by-side local population predictions versus actual historical trends. Also performs coefficient-driven "driver analysis" across different geographic population density clusters.

### Key Directories
- **`feature_selection/`**: Jupyter Notebooks for evaluating different methods (VIF, Lasso, Correlation Matrix) to isolate the most robust predictors.
- **`modelling/`**: Exploratory analyses fine-tuning each of the models and diagnosing prediction bounds.
- **`forward_prediction/`**: Computations generating future population forecasts.
- **`graphing/`**: Plotting workflows tracking model convergence and optimal hyperparameter setups over time.
- **`data/` & `geojson/`**: Sources for structured census statistics and map geometry.
- **`saves/`**: Serialized model results and predictions (`.pickle`), caching expensive modeling tasks.

## Features Profiled
Models evaluate an array of local, thematic indicators tracking spatial transformation. Variables include:
- **Crime Rates**: Detailed counts across dozens of subsets (Burglary, Theft, Vehicle offenses, Violence, etc.).
- **Greenhouse Gas Emissions (GHG)**: Categorized output per sector (Agriculture, Transport, Domestic, Industry).
- **Economic & Educational Quality**: GVA (Gross Value Added) by industry, tax records, unemployment ratios, and GCSE test scores.
- **Infrastructure**: Measured road lengths serving as proxies for connectivity.