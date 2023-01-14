# fraud_detection

automatic pipeline for fraud detection
In this project I implemented fraud detection model using automatic pipeline and Mlflow for model tracking
I also added sample OpenAPI for fraud detection application

### Sample usage for API

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "trans_date_trans_time": "2019-01-01 00:00:18",
  "merchant": "fraud_Rippin, Kub and Mann",
  "amt":454528.6,
  "gender": "M",
  "city": "Malad City",
  "state": "KS",
  "city_pop": 1458,
  "job": "Pathologist",
  "category": "grocery_pos"
}'
```
