import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Clock, AlertCircle } from 'lucide-react';

interface AnalysisStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
}

interface AnalysisProgressProps {
  steps: AnalysisStep[];
  currentStep: string;
}

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({ steps, currentStep }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-success" />;
      case 'running':
        return <Clock className="h-5 w-5 text-primary animate-pulse" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-destructive" />;
      default:
        return <Clock className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default" className="text-xs bg-success text-success-foreground">Completed</Badge>;
      case 'running':
        return <Badge variant="default" className="text-xs">Running</Badge>;
      case 'error':
        return <Badge variant="destructive" className="text-xs">Error</Badge>;
      default:
        return <Badge variant="secondary" className="text-xs">Pending</Badge>;
    }
  };

  const overallProgress = Math.round(
    (steps.filter(step => step.status === 'completed').length / steps.length) * 100
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Analysis Progress
          <span className="text-sm font-normal text-muted-foreground">
            {overallProgress}% Complete
          </span>
        </CardTitle>
        <CardDescription>
          Processing your dataset through our sentiment analysis pipeline
        </CardDescription>
        <Progress value={overallProgress} className="w-full" />
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {steps.map((step) => (
            <div
              key={step.id}
              className={`flex items-center space-x-4 p-4 rounded-lg border transition-all ${
                step.id === currentStep ? 'border-primary bg-primary/5' : 'border-border'
              }`}
            >
              {getStatusIcon(step.status)}
              
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">{step.name}</h4>
                  {getStatusBadge(step.status)}
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {step.description}
                </p>
                
                {step.status === 'running' && (
                  <div className="mt-2">
                    <Progress value={step.progress} className="w-full h-2" />
                    <p className="text-xs text-muted-foreground mt-1">
                      {step.progress}% complete
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};