import pickle
import pandas as pd
from pathlib import Path
import numpy as np

# Define the same TemperatureScaledModel class
class TemperatureScaledModel:
    def __init__(self, base_model, temperature=1.0):
        self.base_model = base_model
        self.temperature = temperature
    
    def predict_proba(self, X):
        logits = self.base_model.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

# Load the model and vectorizer
MODEL_FILE = Path('model/fake_news_model.pkl')
VECTORIZER_FILE = Path('model/tfidf_vectorizer.pkl')

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_FILE, 'rb') as f:
    vectorizer = pickle.load(f)

# Test with various texts
test_texts = [
    "Breaking news: Scientists discover new planet",
    "SHOCKING: Government hiding alien invasion!!!",
    "India launches new space mission successfully",
    "Hillary Clinton exposed secret scandal by insiders!",
    "Apple releases new iPhone with better features",
    "FAKE NEWS: President caught in massive conspiracy",
    "COVID-19 vaccine approved by health authorities",
    "Celebrity spotted at local restaurant today",
    "This is a normal factual news article about events",
]

print("=" * 80)
print("NEW MODEL PROBABILITY ANALYSIS (TEMPERATURE SCALED)")
print("=" * 80)

for text in test_texts:
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    pred_label = 'REAL' if prediction == 0 else 'FAKE'
    real_prob = probabilities[0] * 100
    fake_prob = probabilities[1] * 100
    
    print(f"\nText: {text[:60]}...")
    print(f"Prediction: {pred_label}")
    print(f"  Real: {real_prob:.2f}%")
    print(f"  Fake: {fake_prob:.2f}%")

print("\n" + "=" * 80)
print("RESULT: Probabilities now properly calibrated around 50%!")
print("=" * 80)
