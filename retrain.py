import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from app import db, Detection

# Load existing data
df = pd.read_csv('cleaned_fake_news_dataset.csv')

# Get feedback data from DB
feedback_data = Detection.query.filter(Detection.feedback.isnot(None)).all()
if feedback_data:
    new_texts = [d.input_text for d in feedback_data]
    new_labels = [1 if d.feedback == 'fake' else 0 for d in feedback_data]  # Convert to 0/1

    if new_texts:
        new_df = pd.DataFrame({'text': new_texts, 'label': new_labels})
        df = pd.concat([df, new_df], ignore_index=True)
        print(f'Added {len(new_texts)} feedback samples to training data')

# Preprocess and train
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'New model accuracy: {accuracy:.4f}')

# Save updated model and vectorizer
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print('Model retrained and saved with feedback data.')