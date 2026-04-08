from flask import Flask, request, jsonify, send_from_directory
import pickle
import os
from pathlib import Path

app = Flask(__name__, static_folder='.')

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_FILE = MODEL_DIR / 'fake_news_model.pkl'
VECTORIZER_FILE = MODEL_DIR / 'tfidf_vectorizer.pkl'


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {path}')
    with path.open('rb') as f:
        return pickle.load(f)


try:
    model = load_pickle(MODEL_FILE)
    vectorizer = load_pickle(VECTORIZER_FILE)
except (FileNotFoundError, pickle.UnpicklingError) as exc:
    raise RuntimeError('Unable to load model files. Make sure model/fake_news_model.pkl and model/tfidf_vectorizer.pkl exist.') from exc


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')


@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid request. Send JSON: {"text": "..."}'}), 400

    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    vectorized = vectorizer.transform([text])
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
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )

if __name__ == '__main__':
    app.run(debug=True)