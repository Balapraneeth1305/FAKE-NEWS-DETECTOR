# Fake News Detector

A web-based application that uses machine learning to detect fake news articles. The system analyzes text content and provides a probability score indicating whether the news is likely real or fake.

## Features

- **Real-time Detection**: Input news text and get instant classification results
- **Confidence Scores**: View probability percentages for both real and fake classifications
- **Web API**: Flask backend exposes `/detect` for model predictions
- **Retraining Scripts**: Update the model with new dataset files
- **Docker Support**: Build and run the app in a container

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (recommended)

### Setup
1. Clone or download the project files
2. Navigate to the project directory:
   ```bash
   cd "c:\FAKE NEWS DETECTOR"
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK data if you retrain or preprocess text:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Running the Application

1. Ensure the model files exist in `model/`:
   - `model/fake_news_model.pkl`
   - `model/tfidf_vectorizer.pkl`
2. Start the app:
   ```bash
   python app.py
   ```
3. Open your browser at:
   ```bash
   http://localhost:5000
   ```

## API Usage

Send a POST request to `/detect` with JSON payload:

```json
{
  "text": "Your news article text here"
}
```

Example response:

```json
{
  "result": "Fake",
  "fake_percentage": 78.5,
  "real_percentage": 21.5,
  "accuracy": 78.5
}
```

## Docker Usage

Build the Docker image:

```bash
docker build -t fake-news-detector .
```

Run the container:

```bash
docker run -p 5000:5000 fake-news-detector
```

Then open:

```bash
http://localhost:5000
```

## Pushing to GitHub

1. Create a repository on GitHub.
2. Add the remote:
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   ```
3. Push the current branch:
   ```bash
   git push -u origin main
   ```

## Project Structure

```
├── app.py                 # Flask web application
├── index.html             # Frontend interface
├── styles.css             # CSS styling
├── script.js              # Frontend JavaScript
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build definition
├── .gitignore             # Ignored files for Git
├── model/                 # Trained model files
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── Notebook/              # Jupyter notebooks
│   └── 01_data_validation.ipynb
├── retrain.py             # Model retraining script
├── retrain_with_new_data.py
├── retrain_kaggle.py
└── *.csv                  # Training datasets
```

## Model Details

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: TF-IDF text vectors from news content
- **Training Data**: Several news datasets including FakeNewsNet and BBC news
- **Accuracy**: Depends on dataset quality and may vary

## Limitations

- Trained on English news articles only
- May not perform well on very short or noisy text
- No live fact-checking or external verification
- Basic model approach (not transformer-based)

## Security Notes

- The app is intended for development and demo use
- Do not expose the Flask dev server directly in production
- Pickle files are not safe if loaded from untrusted sources
- Add authentication, input validation, and rate limiting for production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please check the dataset licenses before reuse.
