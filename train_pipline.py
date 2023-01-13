import pandas as pd
import joblib
from sklearn.metrics import f1_score
from urllib.parse import urlparse
from pipeline import pipeline
import hydra
from hydra import utils
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, recall_score, precision_score


@hydra.main(config_name='preprocessing', config_path='config')
def run_training(config):
    """Train the model."""

    current_path = utils.get_original_cwd() + "/"
    # read training data
    data_train = pd.read_csv(current_path + config.dataset.data.Train, encoding=config.dataset.encoding)

    X_train = data_train.drop([config.target.target], axis=1)
    y_train = data_train[[config.target.target]]

    ## test data
    data_test = pd.read_csv(current_path + config.dataset.data.Test, encoding=config.dataset.encoding)

    X_test = data_test.drop([config.target.target], axis=1)
    y_test = data_test[[config.target.target]]

    match_pipe = pipeline(config)
    with mlflow.start_run():
        match_pipe.fit(X_train, y_train)
        pred = match_pipe.predict(X_test)
        mlflow.log_param("max_depth", config.pipeline.decisiontree.parameters.max_depth)
        mlflow.log_param("criterion", config.pipeline.decisiontree.parameters.criterion)
        f1score = f1_score(y_test, y_pred=pred, average='weighted')
        recall = recall_score(y_test, y_pred=pred, average='weighted')
        mlflow.log_metric("f1score", f1score)
        mlflow.log_metric("recall", recall)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(match_pipe, "model", registered_model_name="pipeline_with_tree", artifact_path='.')
        else:
            mlflow.sklearn.log_model(match_pipe, "model")


if __name__ == '__main__':
    run_training()
