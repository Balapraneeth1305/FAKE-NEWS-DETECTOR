import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import os

# Function to load and combine datasets
def load_datasets():
    dataframes = []

    # Load BBC news (real news, label 1)
    if os.path.exists('bbc_news_20220307_20240703.csv'):
        bbc_df = pd.read_csv('bbc_news_20220307_20240703.csv')
        bbc_df['text'] = bbc_df['title'] + ' ' + bbc_df['description']
        bbc_df['label'] = 1  # Real news
        dataframes.append(bbc_df[['text', 'label']])

    # Load fake.csv (fake news, label 0)
    if os.path.exists('fake.csv'):
        fake_df = pd.read_csv('fake.csv')
        fake_df['text'] = fake_df['title'] + ' ' + fake_df['text']
        fake_df['label'] = 0  # Fake news
        dataframes.append(fake_df[['text', 'label']])

    # Load FakeNewsNet.csv
    if os.path.exists('FakeNewsNet.csv'):
        fakenet_df = pd.read_csv('FakeNewsNet.csv')
        fakenet_df = fakenet_df.rename(columns={'real': 'label'})
        fakenet_df['text'] = fakenet_df['title']
        dataframes.append(fakenet_df[['text', 'label']])

    # Load india-news-headlines.csv (real news, label 1)
    if os.path.exists('india-news-headlines.csv'):
        india_df = pd.read_csv('india-news-headlines.csv')
        india_df['text'] = india_df['headline_text']
        india_df['label'] = 1  # Real news
        dataframes.append(india_df[['text', 'label']])

    # Load existing cleaned dataset if it exists
    if os.path.exists('cleaned_fake_news_dataset.csv'):
        existing_df = pd.read_csv('cleaned_fake_news_dataset.csv')
        existing_df = existing_df.rename(columns={'clean_content': 'text'})
        dataframes.append(existing_df[['text', 'label']])

    # Combine all datasets
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates and NaN values
    combined_df = combined_df.dropna(subset=['text', 'label'])
    combined_df = combined_df.drop_duplicates(subset=['text'])

    return combined_df

# Load data
df = load_datasets()

print(f"Total samples: {len(df)}")
print(f"Real news: {len(df[df['label'] == 1])}")
print(f"Fake news: {len(df[df['label'] == 0])}")

# Preprocess and train
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Save updated model and vectorizer
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print('Model retrained and saved with new datasets.')