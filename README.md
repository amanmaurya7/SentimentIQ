# 🧠 SentimentIQ - Professional Sentiment Analysis Platform

A comprehensive sentiment analysis platform that combines a modern React frontend with a powerful Python backend using advanced NLP libraries.

![SentimentIQ](https://img.shields.io/badge/SentimentIQ-AI%20Powered-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![React](https://img.shields.io/badge/React-18-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)

## ✨ Features

### 🤖 **Real AI Analysis**
- **VADER Sentiment Analysis** - Rule-based analysis optimized for social media
- **TextBlob Analysis** - Pattern-based sentiment with polarity scoring
- **Machine Learning Models** - Logistic Regression with TF-IDF vectorization
- **Topic Detection** - Automatic keyword and topic extraction
- **Category Classification** - Intelligent feedback categorization

### 📊 **Professional Dashboard**
- Interactive data visualizations with custom charts
- Real-time analysis progress tracking
- Critical complaint identification and alerts
- Comprehensive sentiment statistics and trends
- Export capabilities for further analysis

### 🔧 **Easy to Use**
- Drag & drop CSV file upload
- Automatic column detection and configuration
- Sample datasets for testing
- Beautiful, responsive design
- Real-time error handling and feedback

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** (for frontend)
- **Python 3.8+** (for backend)
- **pip** (Python package manager)

### Option 1: Quick Start Scripts

**For macOS/Linux:**
```bash
# Start the Python backend
chmod +x start-backend.sh
./start-backend.sh
```

**For Windows:**
```cmd
# Start the Python backend
start-backend.bat
```

**Then in another terminal:**
```bash
# Start the frontend
npm install
npm run dev
```

### Option 2: Manual Setup

**1. Backend Setup:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**2. Frontend Setup:**
```bash
# In project root
npm install
npm run dev
```

### Option 3: Docker Setup
```bash
cd backend
docker-compose up
```

## 📂 Project Structure

```
sentimentiq/
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── pages/           # Application pages
│   │   └── data/            # Sample datasets
├── backend/                 # Python Flask API
│   ├── app.py              # Main API server
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
└── README.md               # This file
```

## 🧪 Analysis Methods

### 1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Rule-based sentiment analysis
- Optimized for social media text
- Provides compound score (-1 to +1) and sentiment breakdown
- Best for: Social media posts, informal text

### 2. **TextBlob**
- Pattern-based sentiment analysis
- Returns polarity (-1 to 1) and subjectivity (0 to 1)
- Good for: General text analysis, blog posts

### 3. **Machine Learning**
- Logistic Regression with TF-IDF features
- Trained on user-provided labeled data (when available)
- Provides prediction probabilities
- Best for: Domain-specific analysis with training data

### 4. **Topic Detection**
- TF-IDF based keyword extraction
- Identifies most important terms and phrases
- Useful for: Understanding main complaint themes

### 5. **Category Classification**
- Rule-based categorization system
- Categories include:
  - Customer Service
  - Product Quality
  - Delivery & Shipping
  - Pricing
  - Technical Issues
  - User Experience

## 📁 CSV Format Requirements

Your CSV file should contain:

**Required:**
- A text column (comments, reviews, feedback, etc.)

**Optional:**
- Existing sentiment labels (for ML training)
- Categories or topics
- Timestamps (for trend analysis)

**Example CSV:**
```csv
id,text,sentiment,category
1,"Great product! Fast delivery and excellent quality.",Positive,Product Quality
2,"Terrible customer service. Had to wait 2 hours.",Negative,Customer Service
3,"Product is okay, nothing special.",Neutral,Product Quality
```

## 🛠️ API Endpoints

The Python backend provides several REST API endpoints:

- `POST /api/analyze` - Analyze sentiment of CSV data
- `POST /api/generate_wordcloud` - Generate word cloud visualization
- `GET /api/health` - Health check endpoint
