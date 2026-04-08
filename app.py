from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__, static_folder='.')

# Load the model and vectorizer
with open('model/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return jsonify({'error': 'Please send a POST request with JSON: {"text": "..."}'}), 405

    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize the text
    vectorized = vectorizer.transform([text])
    # Get prediction probabilities
    proba = model.predict_proba(vectorized)[0]
    fake_proba = proba[1]
    real_proba = proba[0]

    result = 'Fake' if fake_proba > 0.5 else 'Real'

    return jsonify({
        'result': result,
        'fake_percentage': round(fake_proba * 100, 2),
        'real_percentage': round(real_proba * 100, 2),
        'accuracy': round(max(fake_proba, real_proba) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)