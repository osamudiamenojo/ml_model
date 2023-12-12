from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
# create a route that manages user request and does sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]

    # Convert prediction to a JSON-serializable format
    prediction_result = prediction.tolist()  # Convert NumPy array to a Python list
    
    return jsonify({'sentiment': prediction_result})


if __name__ == '__main__':
    app.run()