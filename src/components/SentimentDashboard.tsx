import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SimplePieChart, SimpleBarChart } from '@/components/SimpleCharts';
import { TrendingUp, TrendingDown, AlertTriangle, MessageSquare } from 'lucide-react';

interface SentimentData {
  sentiment: string;
  count: number;
  percentage: number;
}

interface CategoryData {
  category: string;
  positive: number;
  negative: number;
  neutral: number;
}

interface TrendData {
  date: string;
  positive: number;
  negative: number;
  neutral: number;
}

interface SentimentDashboardProps {
  sentimentData: SentimentData[];
  categoryData: CategoryData[];
  trendData: TrendData[];
  totalRecords: number;
  criticalComplaints: any[];
}

const SENTIMENT_COLORS = {
  Positive: '#22c55e',
  Negative: '#ef4444',
  Neutral: '#6b7280'
};

export const SentimentDashboard: React.FC<SentimentDashboardProps> = ({
  sentimentData,
  categoryData,
  trendData,
  totalRecords,
  criticalComplaints
}) => {
  const negativePercentage = sentimentData.find(d => d.sentiment === 'Negative')?.percentage || 0;
  const positivePercentage = sentimentData.find(d => d.sentiment === 'Positive')?.percentage || 0;
  
  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              <div>
                <p className="text-2xl font-bold">{totalRecords.toLocaleString()}</p>
                <p className="text-sm text-muted-foreground">Total Feedback</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-success" />
              <div>
                <p className="text-2xl font-bold text-success">{positivePercentage.toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground">Positive Sentiment</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <TrendingDown className="h-5 w-5 text-destructive" />
              <div>
                <p className="text-2xl font-bold text-destructive">{negativePercentage.toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground">Negative Sentiment</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-warning" />
              <div>
                <p className="text-2xl font-bold text-warning">{criticalComplaints.length}</p>
                <p className="text-sm text-muted-foreground">Critical Issues</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sentiment Distribution Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Overall Sentiment Distribution</CardTitle>
            <CardDescription>Breakdown of sentiment across all feedback</CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center">
            <SimplePieChart
              data={sentimentData.map(item => ({
                name: item.sentiment,
                value: item.count,
                color: SENTIMENT_COLORS[item.sentiment as keyof typeof SENTIMENT_COLORS]
              }))}
              width={400}
              height={350}
            />
          </CardContent>
        </Card>

        {/* Sentiment by Category */}
        <Card>
          <CardHeader>
            <CardTitle>Sentiment by Category</CardTitle>
            <CardDescription>Sentiment breakdown across different categories</CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center">
            <SimpleBarChart
              data={categoryData.map(item => ({
                name: item.category,
                positive: item.positive,
                negative: item.negative,
                neutral: item.neutral
              }))}
              width={500}
              height={350}
            />
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 gap-6">
        {/* Sentiment Trend Over Time */}
        <Card>
          <CardHeader>
            <CardTitle>Sentiment Trends Over Time</CardTitle>
            <CardDescription>Track how sentiment changes over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80 flex items-center justify-center bg-muted/20 rounded-lg">
              <div className="text-center">
                <TrendingUp className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                <p className="text-muted-foreground">Trend visualization coming soon</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {trendData.length} data points available
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Critical Complaints */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-warning" />
            <span>Critical Complaints Requiring Immediate Attention</span>
          </CardTitle>
          <CardDescription>
            Most severe negative feedback identified by AI analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {criticalComplaints.slice(0, 5).map((complaint, index) => (
              <div key={index} className="p-4 border rounded-lg bg-destructive/5 border-destructive/20">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <Badge variant="destructive" className="text-xs">
                        Score: {complaint.score?.toFixed(2) || 'N/A'}
                      </Badge>
                      {complaint.category && (
                        <Badge variant="outline" className="text-xs">
                          {complaint.category}
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm">{complaint.text || complaint.feedback || 'No text available'}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};