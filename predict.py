import hydra
import joblib
import pandas as pd
from hydra import utils
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def make_prediction(input_data, config):
    _pipe_match = joblib.load(filename=utils.to_absolute_path('decisiontree'))

    results = _pipe_match.predict(input_data)

    return results, _pipe_match


@hydra.main(config_name='preprocessing.yaml', config_path='config')
def training(config):
    current_path = utils.get_original_cwd() + "/"

    data_test = pd.read_csv(current_path + config.dataset.data.Test, encoding=config.dataset.encoding)
    X_test = data_test.drop([config.target.target], axis=1)
    y_test = data_test[[config.target.target]]
    pred, pipe = make_prediction(X_test, config)

    # determine mse and rmse
    print('test classifier score(accuracy): {}'.format(pipe.score(X_test, y_test)))
    print('test F1 score: {}'.format(f1_score(y_test, y_pred=pred, average='weighted')))
    print('test Recall: {}'.format(recall_score(y_test, y_pred=pred, average='weighted')))
    print('confusion matrix: {}'.format(
        confusion_matrix(y_test, pred)))


if __name__ == '__main__':
    training()
