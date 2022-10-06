import logging
import mlflow
import mlflow.sklearn
import numpy as np
import os
import pathlib
import sys
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse

base_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(base_path,'data/processed'))  # noqa
sys.path.append(os.path.join(base_path,'scripts'))  # noqa
import ingest_data  # noqa
import train_model  # noqa

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        data = ingest_data.load_housing_data()
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet \
                connection. Error: %s", e
        )

    with mlflow.start_run():

        test_y, predicted_qualities, best_parms, final_model = \
            train_model.fit_direct_data(data, '')

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Best Params :", best_parms)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("Best Params", best_parms)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(final_model, "model", registered_model_name="RandomForestRegressor") # noqa
        else:
            mlflow.sklearn.log_model(final_model, "model")
