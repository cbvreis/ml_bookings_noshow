from flask import Flask, jsonify
import pickle
import pandas as pd

flask_app = Flask(__name__)


@flask_app.route('/')
def index():
    return jsonify({'message': 'Hello, World!'})

@flask_app.route('/model')
def model():
    model = pickle.load(open('models/model.pkl', 'rb'))
    threshold = 0.389
    df = pd.read_csv('data/X_test.csv')
    y_pred_ba = model.predict_proba(df)
    pred2 = pd.Series(y_pred_ba[:, 1]).map(lambda x: 1 if x > threshold else 0)
    return jsonify({'prediction': pred2.values.tolist()})


if __name__ == '__main__':
    flask_app.run(debug=True)
