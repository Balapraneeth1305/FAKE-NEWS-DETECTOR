import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if needed
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# List of datasets to download (Kaggle slugs)
datasets = [
    'clmentbisaillon/fake-and-real-news-dataset',  # Fake and real news dataset
    'jruvika/fake-news-detection',  # Fake news detection
    'rajatkumar30/fake-news',  # Fake news
    'mrisdal/fake-news',  # Fake news
    'emineyetm/fake-news-detection-datasets'  # Fake news detection datasets
]

# Directory to save datasets
data_dir = 'new_datasets'
os.makedirs(data_dir, exist_ok=True)

# Download and unzip datasets
for dataset in datasets:
    print(f'Downloading {dataset}...')
    api.competition_download_files(dataset, path=data_dir, quiet=False)
    # Assuming it's a zip file
    zip_path = os.path.join(data_dir, f'{dataset.split("/")[-1]}.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

# Now, load and combine datasets
# This assumes each dataset has CSV files with 'title', 'text', 'label' columns
# Adjust column names as needed based on actual datasets

dfs = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(root, file))
            # Standardize columns
            if 'title' in df.columns and 'text' in df.columns and 'label' in df.columns:
                df['content'] = df['title'] + " " + df['text']
                df = df[['content', 'label']]
                dfs.append(df)
            elif 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']].rename(columns={'text': 'content'})
                dfs.append(df)
            # Add more conditions if needed

if not dfs:
    print("No suitable datasets found. Please check the downloaded files.")
    exit()

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.dropna(inplace=True)

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

combined_df['content'] = combined_df['content'].apply(preprocess_text)

# Ensure labels are 0 or 1
combined_df['label'] = combined_df['label'].map({'real': 0, 'fake': 1, 0: 0, 1: 1})

# Train-test split
X = combined_df['content']
y = combined_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'New model accuracy: {accuracy:.2f}')

# Save model and vectorizer
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print('Model retrained and saved.')