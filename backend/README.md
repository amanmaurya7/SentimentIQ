# SentimentIQ Backend API

Professional sentiment analysis backend using Python and advanced NLP libraries.

## Features

- **VADER Sentiment Analysis** - Rule-based sentiment analysis optimized for social media text
- **TextBlob Analysis** - Pattern-based sentiment analysis with polarity and subjectivity
- **Machine Learning Models** - Logistic Regression with TF-IDF vectorization
- **Topic Detection** - Automatic topic extraction using TF-IDF
- **Category Classification** - Intelligent categorization of feedback
- **Word Cloud Generation** - Visual representation of most frequent terms
- **Real-time Processing** - Fast analysis of large datasets

## Setup

1. Install Python 3.8+ and pip
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API server:
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

### POST /api/analyze
Analyze sentiment of CSV data with comprehensive insights.

**Request Body:**
```json
{
  "csv_data": [...],
  "text_column": "text",
  "sentiment_column": "sentiment" // optional
}
```

**Response:**
```json
{
  "results": [...],
  "sentiment_data": [...],
  "category_data": [...],
  "critical_complaints": [...],
  "top_topics": [...],
  "trend_data": [...],
  "processing_stats": {...}
}
```

### POST /api/generate_wordcloud
Generate word cloud visualization.

### GET /api/health
Health check endpoint.

## Analysis Methods

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
   - Rule-based sentiment analysis
   - Optimized for social media text
   - Provides compound score and individual sentiment components

2. **TextBlob**
   - Pattern-based sentiment analysis
   - Returns polarity (-1 to 1) and subjectivity (0 to 1)

3. **Machine Learning**
   - Logistic Regression with TF-IDF features
   - Trained on user-provided labeled data
   - Provides prediction probabilities

4. **Topic Detection**
   - TF-IDF based keyword extraction
   - Identifies most important terms and phrases

5. **Category Classification**
   - Rule-based categorization
   - Categories: Customer Service, Product Quality, Delivery, Pricing, Technical Issues