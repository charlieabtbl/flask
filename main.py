import json
from flask import Flask, request, jsonify
from joblib import load
import sklearn
import pickle
app = Flask(__name__)
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]
def loading_model():
    return pickle.loads(load("/Users/charlotteabitbol/Downloads/ITC_download/Assignment Notebooks/model.joblib"))

def scaling():
    return pickle.loads(load("/Users/charlotteabitbol/Downloads/ITC_download/Assignment Notebooks/preprocessing.joblib"))

@app.route('/predict_single')
def single_prediction():
    """ This function receives inputs as parameters (no body), and returns a single prediction as a string / text."""
    logreg = loading_model()
    scaler = scaling()
    parameters = request.args
    X_test = []
    for feature in features:
        X_test.append(float(parameters[feature]))
    X_test_s = scaler.transform([X_test])
    y_pred = logreg.predict(X_test_s)
    return {"input": str(dict(zip(features, X_test))), "output": str(y_pred)}

@app.route('/predict_multiple')
def multiple_prediction():
    logreg = loading_model()
    scaler = scaling()
    body = request.data
    X_test = []
    df = json.loads(body)
    for x in df["data"]:
        X_test.append(list(x.values()))
    X_test_s = scaler.transform(X_test)
    y_pred = logreg.predict(X_test_s)
    return {"output": json.loads(str(y_pred).replace(" ", ","))}


def main():
    app.run()


if __name__ == '__main__':
    main()
