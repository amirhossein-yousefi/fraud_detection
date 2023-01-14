import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from hydra import utils

from Model import FraudFeatures, Labels

# Create the app
app = FastAPI()


def make_prediction(input_data):
    _pipe_match = joblib.load(filename=utils.to_absolute_path('decisiontree'))

    results = _pipe_match.predict(input_data)

    return results, _pipe_match


# Load trained Pipeline
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}


# Define predict function
@app.post('/predict')
def predict(fraud: FraudFeatures):
    data = fraud.dict()
    data = pd.DataFrame(
        [[data['trans_date_trans_time'], data['merchant'], data['amt'], data['gender'], data['city'], data['state'],
          data['city_pop'], data['job'], data['category']]])
    data.columns = ['trans_date_trans_time', 'merchant', 'amt', 'gender', 'city', 'state', 'city_pop', 'job',
                    'category']
    predictions, _ = make_prediction(data)
    return {'prediction': Labels[predictions.tolist()[0]]}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
