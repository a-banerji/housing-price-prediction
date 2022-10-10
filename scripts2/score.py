"""This file stores the model scores and stores the result in the given folder.
By default it loades the model from the pickles folder and stores the output
in the outputs folder.
This file accepts arguments for loading the data from a specific folder,
loads the model from the given arguments and stores it in the given path."""

import argparse
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import sys
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

base_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(os.path.join(base_path,'data/processed'))  # noqa
sys.path.append(os.path.join(base_path,'pickles'))  # noqa
import ingest_data  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--model_load_path",
                    help="Input Path of the model", default='')
parser.add_argument("--data_load_path",
                    help="Data-set path", default='')
parser.add_argument("--output",
                    help="output path", default='')
parser.add_argument("--log-level",
                    help="Choose Log Level from the choice.", default='DEBUG',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
parser.add_argument("--log-path",
                    help="Choose path for log storing", default='')
parser.add_argument("--no-console-log",
                    help="Write Logs to console", default='False')
args = parser.parse_args()

model_load_path = args.model_load_path
input_data = args.data_load_path
output = args.output
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

if model_load_path == '':
    filename = os.path.join(base_path, 'pickles/finalized_model.sav')
    final_model = pickle.load(open(filename, 'rb'))
else:
    filename = os.path.join(model_load_path, 'finalized_model.sav')
    final_model = pickle.load(open(filename, 'rb'))

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

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
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
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
X_test_prepared["population_per_household"] = X_test_prepared["population"]\
                                            / X_test_prepared["households"]

X_test_cat = X_test[['ocean_proximity']]
X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True))


final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

res = pd.DataFrame([final_mse, final_rmse]).T
res.columns = ['MSE', 'RMSE']
res.index = ['Final Model']

if output == '':
    res.to_excel('../outputs/score.xlsx')
else:
    outp = os.path.join(output, 'score.xlsx')
    res.to_excel(outp)
