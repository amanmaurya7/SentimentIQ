import { useState, useCallback } from 'react';

interface AnalysisStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
}

interface VaderResult {
  sentiment: string;
  compound_score: number;
  positive: number;
  negative: number;
  neutral: number;
  confidence: number;
}

interface TextBlobResult {
  sentiment: string;
  polarity: number;
  subjectivity: number;
}

interface MLResult {
  sentiment: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface SentimentResult {
  id: number;
  original_text: string;
  processed_text: string;
  vader: VaderResult;
  textblob: TextBlobResult;
  ml_prediction?: MLResult;
  category: string;
  final_sentiment: 'Positive' | 'Negative' | 'Neutral';
  confidence: number;
}

interface ProcessingStats {
  total_records: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  categories_found: number;
  avg_confidence: number;
}

interface AnalysisResults {
  results: Array<{
    text: string;
    sentiment: 'Positive' | 'Negative' | 'Neutral';
    score: number;
    confidence: number;
    category: string;
    vader?: VaderResult;
    textblob?: TextBlobResult;
    ml_prediction?: MLResult;
  }>;
  sentimentData: Array<{ sentiment: string; count: number; percentage: number }>;
  categoryData: Array<{ category: string; positive: number; negative: number; neutral: number }>;
  trendData: Array<{ date: string; positive: number; negative: number; neutral: number }>;
  criticalComplaints: Array<{ text: string; score: number; category: string; confidence: number }>;
  topTopics?: string[];
  mlResults?: any;
  processingStats?: ProcessingStats;
}

const ANALYSIS_STEPS: AnalysisStep[] = [
  {
    id: 'preprocessing',
    name: 'Data Preprocessing',
    description: 'Cleaning text, removing noise, tokenization, and lemmatization',
    status: 'pending',
    progress: 0
  },
  {
    id: 'sentiment',
    name: 'Sentiment Analysis',
    description: 'Running VADER, TextBlob, and ML models for sentiment classification',
    status: 'pending',
    progress: 0
  },
  {
    id: 'categorization',
    name: 'Topic & Category Detection',
    description: 'Identifying complaint categories and extracting key topics',
    status: 'pending',
    progress: 0
  },
  {
    id: 'insights',
    name: 'Generating Insights',
    description: 'Creating visualizations, identifying critical issues, and computing statistics',
    status: 'pending',
    progress: 0
  }
];

const API_BASE_URL = 'http://localhost:5000/api';

export const useSentimentAnalysis = () => {
  const [steps, setSteps] = useState<AnalysisStep[]>(ANALYSIS_STEPS);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const updateStep = useCallback((stepId: string, updates: Partial<AnalysisStep>) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, ...updates } : step
    ));
  }, []);

  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }, []);

  const runDemoAnalysis = useCallback(async (data: any[], textColumn: string) => {
    setIsDemoMode(true);
    setIsAnalyzing(true);
    setError(null);
    
    // Demo analysis using simple keyword-based approach
    const demoResults = data.map((row, index) => {
      const text = row[textColumn] || '';
      const lowerText = text.toLowerCase();
      
      // Simple sentiment scoring
      let score = 0;
      const positiveWords = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'perfect', 'wonderful'];
      const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor'];
      
      positiveWords.forEach(word => {
        if (lowerText.includes(word)) score += 1;
      });
      negativeWords.forEach(word => {
        if (lowerText.includes(word)) score -= 1;
      });
      
      let sentiment: 'Positive' | 'Negative' | 'Neutral';
      if (score > 0) sentiment = 'Positive';
      else if (score < 0) sentiment = 'Negative';
      else sentiment = 'Neutral';
      
      // Simple categorization
      let category = 'General';
      if (lowerText.includes('service') || lowerText.includes('support')) category = 'Customer Service';
      else if (lowerText.includes('product') || lowerText.includes('quality')) category = 'Product Quality';
      else if (lowerText.includes('delivery') || lowerText.includes('shipping')) category = 'Delivery & Shipping';
      else if (lowerText.includes('price') || lowerText.includes('cost')) category = 'Pricing';
      
      return {
        text,
        sentiment,
        score: score / 3,
        confidence: Math.min(Math.abs(score) / 2, 1),
        category
      };
    });
    
    // Simulate analysis steps
    for (const step of ANALYSIS_STEPS) {
      setCurrentStep(step.id);
      updateStep(step.id, { status: 'running', progress: 0 });
      
      for (let progress = 0; progress <= 100; progress += 25) {
        updateStep(step.id, { progress });
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      updateStep(step.id, { status: 'completed', progress: 100 });
    }
    
    // Generate demo insights
    const sentimentCounts = demoResults.reduce((acc, result) => {
      acc[result.sentiment] = (acc[result.sentiment] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const sentimentData = Object.entries(sentimentCounts).map(([sentiment, count]) => ({
      sentiment,
      count,
      percentage: (count / demoResults.length) * 100
    }));
    
    const categoryMap = new Map<string, { positive: number; negative: number; neutral: number }>();
    demoResults.forEach(result => {
      if (!categoryMap.has(result.category)) {
        categoryMap.set(result.category, { positive: 0, negative: 0, neutral: 0 });
      }
      const counts = categoryMap.get(result.category)!;
      counts[result.sentiment.toLowerCase() as keyof typeof counts]++;
    });
    
    const categoryData = Array.from(categoryMap.entries()).map(([category, counts]) => ({
      category,
      ...counts
    }));
    
    const criticalComplaints = demoResults
      .filter(r => r.sentiment === 'Negative')
      .sort((a, b) => a.score - b.score)
      .slice(0, 5)
      .map(r => ({
        text: r.text,
        score: r.score,
        category: r.category,
        confidence: r.confidence
      }));
    
    const trendData = Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      return {
        date: date.toLocaleDateString(),
        positive: Math.floor(Math.random() * 20) + 5,
        negative: Math.floor(Math.random() * 15) + 2,
        neutral: Math.floor(Math.random() * 10) + 2
      };
    });
    
    setResults({
      results: demoResults,
      sentimentData,
      categoryData,
      trendData,
      criticalComplaints
    });
    
    setIsAnalyzing(false);
    setCurrentStep('');
  }, [updateStep]);

  const runAnalysis = useCallback(async (data: any[], textColumn: string, sentimentColumn?: string) => {
    setIsDemoMode(false);
    setIsAnalyzing(true);
    setResults(null);
    setError(null);
    
    try {
      // Check if backend is running
      const isHealthy = await checkBackendHealth();
      if (!isHealthy) {
        throw new Error('Python backend is not running. Please start the backend server first.');
      }

      // Step 1: Preprocessing
      setCurrentStep('preprocessing');
      updateStep('preprocessing', { status: 'running', progress: 0 });
      
      // Simulate progress for preprocessing
      for (let progress = 0; progress <= 100; progress += 20) {
        updateStep('preprocessing', { progress });
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      updateStep('preprocessing', { status: 'completed', progress: 100 });
      
      // Step 2: Send data to Python backend for analysis
      setCurrentStep('sentiment');
      updateStep('sentiment', { status: 'running', progress: 0 });
      
      const requestData = {
        csv_data: data,
        text_column: textColumn,
        sentiment_column: sentimentColumn
      };
      
      console.log('Sending data to Python backend...');
      
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      updateStep('sentiment', { progress: 50 });
      
      const analysisResults = await response.json();
      console.log('Received results from Python backend:', analysisResults);
      
      updateStep('sentiment', { status: 'completed', progress: 100 });
      
      // Step 3: Categorization (already done by backend)
      setCurrentStep('categorization');
      updateStep('categorization', { status: 'running', progress: 0 });
      
      for (let progress = 0; progress <= 100; progress += 25) {
        updateStep('categorization', { progress });
        await new Promise(resolve => setTimeout(resolve, 150));
      }
      updateStep('categorization', { status: 'completed', progress: 100 });
      
      // Step 4: Generate insights (already done by backend)
      setCurrentStep('insights');
      updateStep('insights', { status: 'running', progress: 0 });
      
      for (let progress = 0; progress <= 100; progress += 33) {
        updateStep('insights', { progress });
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      updateStep('insights', { status: 'completed', progress: 100 });
      
      // Transform backend results to match frontend interface
      const transformedResults: AnalysisResults = {
        results: analysisResults.results.map((result: any) => ({
          text: result.original_text,
          sentiment: result.final_sentiment,
          score: result.vader.compound_score,
          confidence: result.confidence,
          category: result.category,
          vader: result.vader,
          textblob: result.textblob,
          ml_prediction: result.ml_prediction
        })),
        sentimentData: analysisResults.sentiment_data,
        categoryData: analysisResults.category_data,
        trendData: analysisResults.trend_data,
        criticalComplaints: analysisResults.critical_complaints,
        topTopics: analysisResults.top_topics,
        mlResults: analysisResults.ml_results,
        processingStats: analysisResults.processing_stats
      };
      
      setResults(transformedResults);
      
    } catch (error) {
      console.error('Analysis failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setError(errorMessage);
      updateStep(currentStep || 'preprocessing', { status: 'error' });
    } finally {
      setIsAnalyzing(false);
      setCurrentStep('');
    }
  }, [updateStep, currentStep, checkBackendHealth]);

  const resetAnalysis = useCallback(() => {
    setSteps(ANALYSIS_STEPS.map(step => ({ ...step, status: 'pending', progress: 0 })));
    setCurrentStep('');
    setResults(null);
    setIsAnalyzing(false);
    setError(null);
    setIsDemoMode(false);
  }, []);

  return {
    steps,
    currentStep,
    results,
    isAnalyzing,
    error,
    isDemoMode,
    runAnalysis,
    runDemoAnalysis,
    resetAnalysis,
    checkBackendHealth
  };
};