import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, FileText, AlertCircle, CheckCircle, Download } from 'lucide-react';
import { cn } from '@/lib/utils';
import { sampleDatasets, generateSampleCSV } from '@/data/sampleData';

interface FileUploadProps {
  onFileUpload: (file: File, data: any[]) => void;
}

interface UploadState {
  status: 'idle' | 'uploading' | 'success' | 'error';
  message: string;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const [uploadState, setUploadState] = useState<UploadState>({ status: 'idle', message: '' });

  const processCSV = useCallback((file: File) => {
    setUploadState({ status: 'uploading', message: 'Processing your dataset...' });
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        const data = lines.slice(1)
          .filter(line => line.trim())
          .map((line, index) => {
            const values = line.split(',').map(v => v.trim());
            const row: any = { id: index + 1 };
            headers.forEach((header, i) => {
              row[header] = values[i] || '';
            });
            return row;
          });

        if (data.length === 0) {
          throw new Error('No data found in CSV file');
        }

        setUploadState({ status: 'success', message: `Successfully loaded ${data.length} records` });
        onFileUpload(file, data);
      } catch (error) {
        setUploadState({ 
          status: 'error', 
          message: error instanceof Error ? error.message : 'Failed to process file' 
        });
      }
    };
    
    reader.onerror = () => {
      setUploadState({ status: 'error', message: 'Failed to read file' });
    };
    
    reader.readAsText(file);
  }, [onFileUpload]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      processCSV(file);
    }
  }, [processCSV]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
    },
    maxFiles: 1,
  });

  const getStatusIcon = () => {
    switch (uploadState.status) {
      case 'uploading':
        return <Upload className="h-8 w-8 animate-pulse text-primary" />;
      case 'success':
        return <CheckCircle className="h-8 w-8 text-success" />;
      case 'error':
        return <AlertCircle className="h-8 w-8 text-destructive" />;
      default:
        return <FileText className="h-8 w-8 text-muted-foreground" />;
    }
  };

  const getStatusColor = () => {
    switch (uploadState.status) {
      case 'success':
        return 'border-success bg-success/5';
      case 'error':
        return 'border-destructive bg-destructive/5';
      case 'uploading':
        return 'border-primary bg-primary/5';
      default:
        return isDragActive ? 'border-primary bg-primary/5' : 'border-dashed';
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <Card className="overflow-hidden">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Upload Your Dataset</CardTitle>
          <CardDescription>
            Upload a CSV file containing customer feedback or reviews for sentiment analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={cn(
              "border-2 rounded-lg p-8 text-center cursor-pointer transition-all duration-300 hover:border-primary/50",
              getStatusColor()
            )}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center space-y-4">
              {getStatusIcon()}
              
              {uploadState.status === 'idle' ? (
                <>
                  <div>
                    <p className="text-lg font-medium">
                      {isDragActive ? 'Drop your CSV file here' : 'Drag & drop your CSV file here'}
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      or click to browse files
                    </p>
                  </div>
                  <Button variant="outline" size="sm">
                    Browse Files
                  </Button>
                </>
              ) : (
                <div>
                  <p className="text-lg font-medium">{uploadState.message}</p>
                  {uploadState.status === 'error' && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={() => setUploadState({ status: 'idle', message: '' })}
                    >
                      Try Again
                    </Button>
                  )}
                </div>
              )}
            </div>
          </div>
          
          <div className="mt-6 space-y-4">
            <div className="text-sm text-muted-foreground">
              <h4 className="font-medium mb-2">Expected CSV Format:</h4>
              <ul className="space-y-1">
                <li>• Required columns: text/content column for feedback</li>
                <li>• Optional: sentiment labels, categories, timestamps</li>
                <li>• Maximum file size: 10MB</li>
                <li>• Supported format: CSV (.csv)</li>
              </ul>
            </div>
            
            <div className="border-t pt-4">
              <h4 className="font-medium mb-3">Try Sample Datasets</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(sampleDatasets).map(([key, dataset]) => (
                  <div key={key} className="p-3 border rounded-lg">
                    <h5 className="font-medium text-sm">{dataset.name}</h5>
                    <p className="text-xs text-muted-foreground mb-2">{dataset.description}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const csvContent = generateSampleCSV(key as keyof typeof sampleDatasets);
                        const blob = new Blob([csvContent], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${key}_sample.csv`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                    >
                      <Download className="h-3 w-3 mr-1" />
                      Download
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};