import pickle
import pandas as pd
from pathlib import Path

# Load the current model and vectorizer
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
    "UNBELIEVABLE: Man discovers fountain of youth!",
    "This is a normal factual news article about events",
]

print("=" * 80)
print("CURRENT MODEL PROBABILITY ANALYSIS")
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

# Check model coefficients to understand what features it learned
print("\n" + "=" * 80)
print("MODEL ANALYSIS")
print("=" * 80)
print(f"Number of features: {len(model.coef_[0])}")
print(f"Model intercept: {model.intercept_[0]:.4f}")
print(f"Coef min: {model.coef_[0].min():.4f}")
print(f"Coef max: {model.coef_[0].max():.4f}")
print(f"Coef mean: {model.coef_[0].mean():.4f}")
print(f"Coef std: {model.coef_[0].std():.4f}")

# Find top features indicating fake vs real
feature_names = vectorizer.get_feature_names_out()
top_fake_indices = model.coef_[0].argsort()[-5:][::-1]
top_real_indices = model.coef_[0].argsort()[:5]

print("\nTop features for FAKE (highest coefficients):")
for idx in top_fake_indices:
    print(f"  {feature_names[idx]}: {model.coef_[0][idx]:.4f}")

print("\nTop features for REAL (lowest coefficients):")
for idx in top_real_indices:
    print(f"  {feature_names[idx]}: {model.coef_[0][idx]:.4f}")
