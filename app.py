from flask import Flask, request, jsonify, send_from_directory
import pickle
import os
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__, static_folder='.')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_FILE = MODEL_DIR / 'fake_news_model.pkl'
VECTORIZER_FILE = MODEL_DIR / 'tfidf_vectorizer.pkl'

# Database model for feedback
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.String(10))  # 'real' or 'fake'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'input_text': self.input_text,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'feedback': self.feedback,
            'timestamp': self.timestamp.isoformat()
        }


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


@app.route('/feedback/<int:detection_id>', methods=['POST'])
def submit_feedback(detection_id):
    try:
        detection = Detection.query.get_or_404(detection_id)
        data = request.get_json()

        if 'feedback' not in data or data['feedback'] not in ['real', 'fake']:
            return jsonify({'error': 'Invalid feedback. Must be "real" or "fake"'}), 400

        detection.feedback = data['feedback']
        db.session.commit()

        return jsonify({'message': 'Feedback submitted successfully'})

    except Exception as e:
        app.logger.error(f'Error in feedback endpoint: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/history', methods=['GET'])
def get_history():
    try:
        limit = min(int(request.args.get('limit', 10)), 50)  # Max 50 items
        detections = Detection.query.order_by(Detection.timestamp.desc()).limit(limit).all()

        return jsonify({
            'history': [d.to_dict() for d in detections],
            'total': len(detections)
        })

    except Exception as e:
        app.logger.error(f'Error in history endpoint: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json(silent=True)
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid request. Send JSON: {"text": "..."}'}), 400

        text = data.get('text', '').strip()

        # Input validation
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400

        if len(text) > 10000:
            return jsonify({'error': 'Text too long (maximum 10,000 characters)'}), 400

        # Make prediction
        vectorized = vectorizer.transform([text])
        proba = model.predict_proba(vectorized)[0]
        fake_proba = proba[1]
        real_proba = proba[0]
        result = 'Fake' if fake_proba > 0.5 else 'Real'

        # Save to database
        detection = Detection(
            input_text=text,
            prediction=result,
            confidence=round(max(fake_proba, real_proba) * 100, 2)
        )
        db.session.add(detection)
        db.session.commit()

        return jsonify({
            'result': result,
            'fake_percentage': round(fake_proba * 100, 2),
            'real_percentage': round(real_proba * 100, 2),
            'accuracy': round(max(fake_proba, real_proba) * 100, 2),
            'detection_id': detection.id
        })

    except Exception as e:
        app.logger.error(f'Error in detect endpoint: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )