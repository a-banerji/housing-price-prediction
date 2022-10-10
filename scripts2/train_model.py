"""
This file loads the data from the data folder and runs multiple
models and hypertunings and finally stores the final model in the
pickles folder and model scores in the outputs folder.
"""

import argparse
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import optuna
import pickle
import sys
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

def fit_direct_data(housing, model_output_path='',
                    base_path=pathlib.Path(__file__).parent.parent.resolve()):

    train_set, test_set = train_test_split(housing, test_size=0.2,
                                           random_state=42)

    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size=0.2,
                                           random_state=42)

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] \
        / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] \
        / compare_props["Overall"] - 100

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"]\
        / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]\
        / housing["total_rooms"]
    housing["population_per_household"] = housing["population"]\
        / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop('ocean_proximity', axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"]\
        / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"]\
        / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"]\
        / housing_tr["households"]

    housing_cat = housing[['ocean_proximity']]
    housing_prepared = housing_tr.join(
                        pd.get_dummies(housing_cat, drop_first=True))
    
    def objective(trial,housing_prepared=housing_prepared, housing_labels=housing_labels):
        
        global best_parms, grid_search
        n_estimators=trial.suggest_int("n_estimators", 3, 30)
        max_features=trial.suggest_int("max_features", 2, 8)
        
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {'n_estimators': [n_estimators], 'max_features': [max_features]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [n_estimators],
             'max_features': [max_features]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)

        best_parms = grid_search.best_params_
        cvres = grid_search.cv_results_
        ss=[]
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
            ss.append(np.sqrt(-mean_score))
        print(np.mean(ss))
        return np.mean(ss)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, housing_prepared=housing_prepared, housing_labels=housing_labels), n_trials=50)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    if model_output_path == '':
        # checks for user arguments, is empty stores default.
        filename = os.path.join(base_path, 'pickles/finalized_model.sav')
    else:
        filename = os.path.join(model_output_path, 'finalized_model.sav')

    pickle.dump(final_model, open(filename, 'wb'))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop('ocean_proximity', axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared,
        columns=X_test_num.columns, index=X_test.index)
    X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"]\
        / X_test_prepared["households"]
    X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"]\
        / X_test_prepared["total_rooms"]
    X_test_prepared["population_per_household"] = \
        X_test_prepared["population"] / X_test_prepared["households"]

    X_test_cat = X_test[['ocean_proximity']]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True))

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("MSE: ", final_mse)
    print("RMSE: ", final_rmse)

    return y_test, final_predictions, best_parms, final_model


if __name__ == "__main__":

    base_path = pathlib.Path(__file__).parent.parent.resolve()
    sys.path.append(os.path.join(base_path,'data/processed'))  # noqa
    import ingest_data  # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",
                        help="Input Path of the data", default='')
    parser.add_argument("--model_output_path",
                        help="Model save path", default='')
    parser.add_argument("--log-level",
                        help="Choose Log Level from the choice.",
                        default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'])
    parser.add_argument("--log-path",
                        help="Choose path for log storing", default='')
    parser.add_argument("--no-console-log",
                        help="Write Logs to console", default='False')
    args = parser.parse_args()

    model_output_path = args.model_output_path
    input_data = args.input_data
    logLevel = args.log_level
    log_path = args.log_path
    no_console_log = args.no_console_log

    logger = logging.getLogger()

    if logLevel.upper() == 'CRITICAL':
        logging.basicConfig(level=logging.CRITICAL)
    if logLevel.upper() == 'ERROR':
        logging.basicConfig(level=logging.ERROR)
    if logLevel.upper() == 'INFO':
        logging.basicConfig(level=logging.INFO)
    if logLevel.upper() == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)

    if log_path != '':
        l1 = os.path.join(log_path, 'Logs.log')
        logging.basicConfig(filename=l1)

    if no_console_log:
        logging.disable(logging.DEBUG)

    if input_data == '':
        # checks for arguments, if empty calls the data-generation script.
        housing = ingest_data.load_housing_data()
    else:
        housing = pd.read_csv(input_data)

    if input_data == '':
        # checks for arguments, if empty calls the data-generation script.
        housing = ingest_data.load_housing_data()
    else:
        housing = pd.read_csv(input_data)

    fit_direct_data(housing, model_output_path)
