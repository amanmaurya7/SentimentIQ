from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import io
import base64

# NLP Libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans

# Data Processing
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

app = Flask(__name__)
CORS(app)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize global analyzers
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class SentimentAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.ml_model = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs, mentions, hashtags, special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return {
            'sentiment': sentiment,
            'compound_score': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'confidence': abs(compound)
        }
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def train_ml_model(self, texts, labels):
        """Train a machine learning model for sentiment analysis"""
        if len(set(labels)) < 2:
            return False
            
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        self.ml_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict_ml_sentiment(self, text):
        """Predict sentiment using trained ML model"""
        if not self.is_trained:
            return None
            
        processed_text = self.preprocess_text(text)
        X = self.tfidf_vectorizer.transform([processed_text])
        
        prediction = self.ml_model.predict(X)[0]
        probabilities = self.ml_model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.ml_model.classes_, probabilities))
        }
    
    def detect_topics(self, texts, n_topics=5):
        """Detect topics using keyword extraction"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Use TF-IDF to find important words
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(processed_texts)
        
        # Get feature names and their importance
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = X.sum(axis=0).A1
        
        # Get top keywords
        top_indices = tfidf_scores.argsort()[-n_topics:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        
        return top_keywords
    
    def categorize_feedback(self, text):
        """Categorize feedback based on keywords"""
        text_lower = text.lower()
        
        categories = {
            'Customer Service': ['service', 'staff', 'support', 'help', 'representative', 'agent', 'response'],
            'Product Quality': ['quality', 'product', 'defect', 'broken', 'durability', 'material', 'build'],
            'Delivery & Shipping': ['delivery', 'shipping', 'package', 'arrived', 'late', 'fast', 'time'],
            'Pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth'],
            'Technical Issues': ['website', 'app', 'technical', 'bug', 'error', 'crash', 'loading'],
            'User Experience': ['easy', 'difficult', 'confusing', 'intuitive', 'design', 'interface']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'General'

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        csv_data = data.get('csv_data', [])
        text_column = data.get('text_column', '')
        existing_sentiment_column = data.get('sentiment_column', '')
        
        if not csv_data or not text_column:
            return jsonify({'error': 'Missing required data'}), 400
        
        df = pd.DataFrame(csv_data)
        
        # Phase 1: Data Preprocessing
        texts = df[text_column].fillna('').astype(str).tolist()
        processed_texts = [analyzer.preprocess_text(text) for text in texts]
        
        results = []
        
        # Phase 2: Sentiment Analysis
        for i, (original_text, processed_text) in enumerate(zip(texts, processed_texts)):
            # VADER Analysis
            vader_result = analyzer.analyze_sentiment_vader(original_text)
            
            # TextBlob Analysis
            textblob_result = analyzer.analyze_sentiment_textblob(original_text)
            
            # Categorization
            category = analyzer.categorize_feedback(original_text)
            
            result = {
                'id': i + 1,
                'original_text': original_text,
                'processed_text': processed_text,
                'vader': vader_result,
                'textblob': textblob_result,
                'category': category,
                'final_sentiment': vader_result['sentiment'],  # Use VADER as primary
                'confidence': vader_result['confidence']
            }
            results.append(result)
        
        # Train ML model if sentiment labels exist
        ml_results = None
        if existing_sentiment_column and existing_sentiment_column in df.columns:
            existing_labels = df[existing_sentiment_column].fillna('').tolist()
            valid_labels = [label for label in existing_labels if label in ['Positive', 'Negative', 'Neutral']]
            
            if len(valid_labels) > 10:  # Need minimum data to train
                ml_results = analyzer.train_ml_model(texts, existing_labels)
                
                # Add ML predictions
                for i, result in enumerate(results):
                    ml_pred = analyzer.predict_ml_sentiment(texts[i])
                    if ml_pred:
                        result['ml_prediction'] = ml_pred
        
        # Phase 3: Generate Insights
        sentiments = [r['final_sentiment'] for r in results]
        sentiment_counts = Counter(sentiments)
        total_count = len(results)
        
        sentiment_data = [
            {
                'sentiment': sentiment,
                'count': count,
                'percentage': (count / total_count) * 100
            }
            for sentiment, count in sentiment_counts.items()
        ]
        
        # Category analysis
        categories = [r['category'] for r in results]
        category_sentiment = {}
        for result in results:
            cat = result['category']
            sent = result['final_sentiment']
            
            if cat not in category_sentiment:
                category_sentiment[cat] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            category_sentiment[cat][sent] += 1
        
        category_data = [
            {
                'category': cat,
                'positive': counts['Positive'],
                'negative': counts['Negative'],
                'neutral': counts['Neutral']
            }
            for cat, counts in category_sentiment.items()
        ]
        
        # Critical complaints (most negative)
        negative_results = [r for r in results if r['final_sentiment'] == 'Negative']
        critical_complaints = sorted(
            negative_results,
            key=lambda x: x['vader']['compound_score']
        )[:10]
        
        critical_data = [
            {
                'text': complaint['original_text'],
                'score': complaint['vader']['compound_score'],
                'category': complaint['category'],
                'confidence': complaint['confidence']
            }
            for complaint in critical_complaints
        ]
        
        # Topic detection
        top_topics = analyzer.detect_topics(texts, n_topics=10)
        
        # Generate trend data (mock for now, would use timestamps if available)
        trend_data = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=6-i)).strftime('%Y-%m-%d')
            # Simulate daily sentiment distribution
            daily_positive = np.random.poisson(sentiment_counts.get('Positive', 0) / 7)
            daily_negative = np.random.poisson(sentiment_counts.get('Negative', 0) / 7)
            daily_neutral = np.random.poisson(sentiment_counts.get('Neutral', 0) / 7)
            
            trend_data.append({
                'date': date,
                'positive': int(daily_positive),
                'negative': int(daily_negative),
                'neutral': int(daily_neutral)
            })
        
        response = {
            'results': results,
            'sentiment_data': sentiment_data,
            'category_data': category_data,
            'critical_complaints': critical_data,
            'top_topics': top_topics,
            'trend_data': trend_data,
            'ml_results': ml_results,
            'total_analyzed': len(results),
            'processing_stats': {
                'total_records': len(results),
                'positive_count': sentiment_counts.get('Positive', 0),
                'negative_count': sentiment_counts.get('Negative', 0),
                'neutral_count': sentiment_counts.get('Neutral', 0),
                'categories_found': len(category_sentiment),
                'avg_confidence': np.mean([r['confidence'] for r in results])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Sentiment analysis API is running'})

@app.route('/api/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.json
        texts = data.get('texts', [])
        sentiment_filter = data.get('sentiment', 'all')  # 'positive', 'negative', 'neutral', or 'all'
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        # Filter texts by sentiment if specified
        if sentiment_filter != 'all':
            # This would require sentiment analysis results
            pass
        
        # Combine all texts
        combined_text = ' '.join([analyzer.preprocess_text(text) for text in texts])
        
        if not combined_text.strip():
            return jsonify({'error': 'No valid text after preprocessing'}), 400
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        # Convert to base64 image
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'wordcloud_image': f'data:image/png;base64,{img_str}',
            'word_frequencies': dict(wordcloud.words_)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Sentiment Analysis API...")
    print("Available endpoints:")
    print("- POST /api/analyze - Analyze sentiment of CSV data")
    print("- POST /api/generate_wordcloud - Generate word cloud")
    print("- GET /api/health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5000)