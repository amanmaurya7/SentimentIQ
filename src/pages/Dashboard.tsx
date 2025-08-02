import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { FileUpload } from '@/components/FileUpload';
import { DataPreview } from '@/components/DataPreview';
import { AnalysisProgress } from '@/components/AnalysisProgress';
import { SentimentDashboard } from '@/components/SentimentDashboard';
import { useSentimentAnalysis } from '@/hooks/useSentimentAnalysis';
import { 
  Brain, 
  Download, 
  RotateCcw, 
  BarChart3, 
  FileText, 
  Settings,
  Database,
  TrendingUp,
  AlertCircle
} from 'lucide-react';

interface UploadedFile {
  file: File;
  data: any[];
}

export const Dashboard: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<{
    text: string;
    sentiment?: string;
    category?: string;
  }>({ text: '' });
  const [activeTab, setActiveTab] = useState('upload');
  
  const { steps, currentStep, results, isAnalyzing, error, isDemoMode, runAnalysis, runDemoAnalysis, resetAnalysis } = useSentimentAnalysis();

  const handleFileUpload = (file: File, data: any[]) => {
    setUploadedFile({ file, data });
    setActiveTab('preview');
    
    // Auto-detect text column
    const columns = Object.keys(data[0] || {});
    const textColumn = columns.find(col => 
      col.toLowerCase().includes('text') || 
      col.toLowerCase().includes('comment') || 
      col.toLowerCase().includes('feedback') ||
      col.toLowerCase().includes('review')
    ) || columns[0];
    
    setSelectedColumns({ text: textColumn });
  };

  const handleStartAnalysis = async () => {
    if (!uploadedFile || !selectedColumns.text) return;
    
    setActiveTab('analysis');
    await runAnalysis(uploadedFile.data, selectedColumns.text, selectedColumns.sentiment);
  };

  const handleStartDemoAnalysis = async () => {
    if (!uploadedFile || !selectedColumns.text) return;
    
    setActiveTab('analysis');
    await runDemoAnalysis(uploadedFile.data, selectedColumns.text);
  };

  const handleNewAnalysis = () => {
    setUploadedFile(null);
    setSelectedColumns({ text: '' });
    setActiveTab('upload');
    resetAnalysis();
  };

  const handleExportResults = () => {
    if (!results) return;
    
    const csvContent = [
      ['Text', 'Sentiment', 'Score', 'Confidence', 'Category'],
      ...results.results.map(r => [
        r.text,
        r.sentiment,
        r.score.toString(),
        r.confidence.toString(),
        r.category || ''
      ])
    ].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentiment_analysis_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <Brain className="h-8 w-8 text-primary" />
                <div>
                  <h1 className="text-2xl font-bold">SentimentIQ</h1>
                  <p className="text-sm text-muted-foreground">AI-Powered Sentiment Analysis Platform</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              {results && (
                <Button variant="outline" size="sm" onClick={handleExportResults}>
                  <Download className="h-4 w-4 mr-2" />
                  Export Results
                </Button>
              )}
              {uploadedFile && (
                <Button variant="outline" size="sm" onClick={handleNewAnalysis}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  New Analysis
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="upload" className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span>Upload Data</span>
            </TabsTrigger>
            <TabsTrigger value="preview" disabled={!uploadedFile} className="flex items-center space-x-2">
              <Database className="h-4 w-4" />
              <span>Data Preview</span>
            </TabsTrigger>
            <TabsTrigger value="analysis" disabled={!uploadedFile} className="flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <span>Analysis</span>
            </TabsTrigger>
            <TabsTrigger value="results" disabled={!results} className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Dashboard</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold mb-2">Welcome to SentimentIQ</h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Upload your customer feedback data and get instant AI-powered sentiment analysis 
                with actionable insights and beautiful visualizations.
              </p>
            </div>
            
            <FileUpload onFileUpload={handleFileUpload} />
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
              <Card>
                <CardHeader>
                  <Brain className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>AI-Powered Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Advanced NLP models analyze sentiment with high accuracy and provide confidence scores.
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <TrendingUp className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>Real-time Insights</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Get instant visualizations and identify critical issues that need immediate attention.
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <Download className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>Export Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Download comprehensive reports and analysis results for further processing.
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="preview" className="space-y-6">
            {uploadedFile && (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Data Preview</h2>
                    <p className="text-muted-foreground">
                      Review your data and configure analysis settings
                    </p>
                  </div>
                  <div className="flex space-x-3">
                    <Button 
                      onClick={handleStartAnalysis} 
                      disabled={!selectedColumns.text}
                      variant="hero"
                      size="lg"
                    >
                      <Brain className="h-5 w-5 mr-2" />
                      Real AI Analysis
                    </Button>
                    <Button 
                      onClick={handleStartDemoAnalysis} 
                      disabled={!selectedColumns.text}
                      variant="outline"
                      size="lg"
                    >
                      Try Demo Mode
                    </Button>
                  </div>
                </div>

                {/* Column Selection */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Settings className="h-5 w-5" />
                      <span>Analysis Configuration</span>
                    </CardTitle>
                    <CardDescription>
                      Select the columns to use for sentiment analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="text-sm font-medium">Text Column (Required)</label>
                        <select 
                          className="w-full mt-1 p-2 border rounded-md"
                          value={selectedColumns.text}
                          onChange={(e) => setSelectedColumns(prev => ({ ...prev, text: e.target.value }))}
                        >
                          <option value="">Select column...</option>
                          {Object.keys(uploadedFile.data[0] || {}).map(col => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="text-sm font-medium">Existing Sentiment (Optional)</label>
                        <select 
                          className="w-full mt-1 p-2 border rounded-md"
                          value={selectedColumns.sentiment || ''}
                          onChange={(e) => setSelectedColumns(prev => ({ ...prev, sentiment: e.target.value }))}
                        >
                          <option value="">None</option>
                          {Object.keys(uploadedFile.data[0] || {}).map(col => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="text-sm font-medium">Category (Optional)</label>
                        <select 
                          className="w-full mt-1 p-2 border rounded-md"
                          value={selectedColumns.category || ''}
                          onChange={(e) => setSelectedColumns(prev => ({ ...prev, category: e.target.value }))}
                        >
                          <option value="">None</option>
                          {Object.keys(uploadedFile.data[0] || {}).map(col => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <DataPreview data={uploadedFile.data} fileName={uploadedFile.file.name} />
              </>
            )}
          </TabsContent>

          <TabsContent value="analysis" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Analysis in Progress</h2>
              <p className="text-muted-foreground">
                Your data is being processed through our AI sentiment analysis pipeline
              </p>
            </div>
            
            <AnalysisProgress steps={steps} currentStep={currentStep} />
            
            {error && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="h-6 w-6 text-destructive mt-0.5" />
                  <div className="flex-1">
                    <h3 className="font-semibold text-destructive">Analysis Failed</h3>
                    <p className="text-sm text-destructive/80 mt-1">{error}</p>
                    
                    {error.includes('backend') && (
                      <div className="mt-4 p-4 bg-card border rounded-lg">
                        <h4 className="font-medium mb-2">🐍 How to Start the Python Backend:</h4>
                        <div className="space-y-2 text-sm">
                          <div className="font-mono bg-muted p-2 rounded">
                            <div>cd backend</div>
                            <div>pip install -r requirements.txt</div>
                            <div>python app.py</div>
                          </div>
                          <p className="text-muted-foreground">
                            The backend will start on <code className="bg-muted px-1 rounded">http://localhost:5000</code>
                          </p>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex space-x-2 mt-4">
                      <Button variant="outline" size="sm" onClick={handleStartDemoAnalysis}>
                        Try Demo Mode
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
                        Retry Real Analysis
                      </Button>
                      <Button variant="outline" size="sm" onClick={handleNewAnalysis}>
                        Start Over
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {results && (
              <div className="text-center">
                <Badge variant="default" className={isDemoMode ? "bg-warning text-warning-foreground mb-4" : "bg-success text-success-foreground mb-4"}>
                  {isDemoMode ? 'Demo Analysis Complete!' : 'Real AI Analysis Complete!'}
                </Badge>
                <div>
                  <Button 
                    onClick={() => setActiveTab('results')} 
                    variant="hero" 
                    size="lg"
                  >
                    <BarChart3 className="h-5 w-5 mr-2" />
                    View Dashboard
                  </Button>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            {results && (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Sentiment Analysis Dashboard</h2>
                    <p className="text-muted-foreground">
                      Comprehensive insights from your customer feedback data
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="default" className="bg-success text-success-foreground">
                      {results.results.length} Records Analyzed
                    </Badge>
                  </div>
                </div>
                
                <SentimentDashboard
                  sentimentData={results.sentimentData}
                  categoryData={results.categoryData}
                  trendData={results.trendData}
                  totalRecords={results.results.length}
                  criticalComplaints={results.criticalComplaints}
                />
              </>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};