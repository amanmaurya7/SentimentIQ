import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { FileText, Database, BarChart3 } from 'lucide-react';

interface DataPreviewProps {
  data: any[];
  fileName: string;
}

export const DataPreview: React.FC<DataPreviewProps> = ({ data, fileName }) => {
  const columns = data.length > 0 ? Object.keys(data[0]) : [];
  const previewData = data.slice(0, 10); // Show first 10 rows
  
  const getColumnType = (column: string) => {
    const sampleValue = data[0]?.[column];
    if (typeof sampleValue === 'number') return 'number';
    if (column.toLowerCase().includes('date')) return 'date';
    if (column.toLowerCase().includes('sentiment')) return 'sentiment';
    if (column.toLowerCase().includes('category')) return 'category';
    return 'text';
  };

  const getColumnIcon = (type: string) => {
    switch (type) {
      case 'number':
        return <BarChart3 className="h-4 w-4" />;
      case 'sentiment':
        return <Badge variant="outline" className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Dataset Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-primary" />
              <div>
                <p className="text-2xl font-bold">{data.length.toLocaleString()}</p>
                <p className="text-sm text-muted-foreground">Total Records</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <FileText className="h-5 w-5 text-primary" />
              <div>
                <p className="text-2xl font-bold">{columns.length}</p>
                <p className="text-sm text-muted-foreground">Columns</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              <div>
                <p className="text-2xl font-bold">{fileName}</p>
                <p className="text-sm text-muted-foreground">Dataset</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Column Information */}
      <Card>
        <CardHeader>
          <CardTitle>Column Information</CardTitle>
          <CardDescription>
            Overview of your dataset structure and column types
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {columns.map((column) => {
              const type = getColumnType(column);
              return (
                <div
                  key={column}
                  className="flex items-center space-x-3 p-3 rounded-lg border bg-card"
                >
                  {getColumnIcon(type)}
                  <div className="flex-1">
                    <p className="font-medium">{column}</p>
                    <p className="text-sm text-muted-foreground capitalize">{type}</p>
                  </div>
                  <Badge variant="secondary" className="text-xs">
                    {type}
                  </Badge>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Data Preview Table */}
      <Card>
        <CardHeader>
          <CardTitle>Data Preview</CardTitle>
          <CardDescription>
            First 10 rows of your dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {columns.map((column) => (
                    <TableHead key={column} className="font-medium">
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {previewData.map((row, index) => (
                  <TableRow key={index}>
                    {columns.map((column) => (
                      <TableCell key={column} className="max-w-xs">
                        <div className="truncate" title={row[column]}>
                          {row[column]}
                        </div>
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};