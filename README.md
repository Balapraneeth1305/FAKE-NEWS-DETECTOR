# Fake News Detector

A web-based application that uses machine learning to detect fake news articles. The system analyzes text content and provides a probability score indicating whether the news is likely real or fake.

## Features

- **Real-time Detection**: Input news text and get instant classification results
- **Confidence Scores**: View probability percentages for both real and fake classifications
- **Text Analysis**: Automatic detection of sensational language patterns
- **Retraining Capability**: Update the model with new data and user feedback
- **Responsive Web Interface**: Modern, dark-themed UI that works on desktop and mobile

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK data (required for text preprocessing):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### Running the Application
1. Ensure the model files (`model/fake_news_model.pkl` and `model/tfidf_vectorizer.pkl`) exist
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://localhost:5000`

### API Usage
Send a POST request to `/detect` with JSON payload:
```json
{
  "text": "Your news article text here"
}
```

Response:
```json
{
  "result": "Fake",
  "fake_percentage": 78.5,
  "real_percentage": 21.5,
  "accuracy": 78.5
}
```

### Retraining the Model
Use the provided retraining scripts to update the model with new data:
- `retrain.py`: Retrains with existing data plus database feedback
- `retrain_with_new_data.py`: Retrains with new CSV data
- `retrain_kaggle.py`: Downloads and retrains with Kaggle datasets

## Project Structure

```
├── app.py                 # Flask web application
├── index.html            # Frontend interface
├── styles.css            # CSS styling
├── script.js             # Frontend JavaScript
├── requirements.txt      # Python dependencies
├── model/                # Trained model files
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── Notebook/             # Jupyter notebooks
│   └── 01_data_validation.ipynb
├── retrain*.py           # Model retraining scripts
└── *.csv                 # Training datasets
```

## Model Details

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: 5000 most frequent words
- **Training Data**: Multiple news datasets including FakeNewsNet and BBC news
- **Accuracy**: Approximately 85-90% (varies by dataset)

## Limitations

- Model trained on English news articles only
- May not perform well on very short texts or non-news content
- Accuracy depends on training data quality and quantity
- No real-time fact-checking against external sources
- Basic NLP approach (not using modern transformers like BERT)

## Security Notes

- This is a development version with debug mode enabled
- For production use, disable debug mode and implement proper security measures
- Model files use pickle (insecure for untrusted data)
- No input validation or rate limiting implemented

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please check individual dataset licenses for usage rights.