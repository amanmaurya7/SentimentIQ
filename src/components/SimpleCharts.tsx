import React from 'react';

interface PieChartData {
  name: string;
  value: number;
  color: string;
}

interface BarChartData {
  name: string;
  positive: number;
  negative: number;
  neutral: number;
}

interface SimplePieChartProps {
  data: PieChartData[];
  width?: number;
  height?: number;
}

interface SimpleBarChartProps {
  data: BarChartData[];
  width?: number;
  height?: number;
}

export const SimplePieChart: React.FC<SimplePieChartProps> = ({ 
  data, 
  width = 400, 
  height = 400 
}) => {
  const radius = Math.min(width, height) / 3;
  const centerX = width / 2;
  const centerY = height / 2;
  
  const total = data.reduce((sum, item) => sum + item.value, 0);
  let currentAngle = 0;
  
  const slices = data.map((item, index) => {
    const percentage = (item.value / total) * 100;
    const angle = (item.value / total) * 2 * Math.PI;
    const startAngle = currentAngle;
    const endAngle = currentAngle + angle;
    
    const x1 = centerX + radius * Math.cos(startAngle);
    const y1 = centerY + radius * Math.sin(startAngle);
    const x2 = centerX + radius * Math.cos(endAngle);
    const y2 = centerY + radius * Math.sin(endAngle);
    
    const largeArcFlag = angle > Math.PI ? 1 : 0;
    
    const pathData = [
      `M ${centerX} ${centerY}`,
      `L ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ');
    
    currentAngle += angle;
    
    // Label position
    const labelAngle = startAngle + angle / 2;
    const labelRadius = radius * 0.7;
    const labelX = centerX + labelRadius * Math.cos(labelAngle);
    const labelY = centerY + labelRadius * Math.sin(labelAngle);
    
    return {
      pathData,
      color: item.color,
      labelX,
      labelY,
      percentage: percentage.toFixed(1),
      name: item.name
    };
  });
  
  return (
    <div className="flex flex-col items-center">
      <svg width={width} height={height} className="overflow-visible">
        {slices.map((slice, index) => (
          <g key={index}>
            <path
              d={slice.pathData}
              fill={slice.color}
              stroke="white"
              strokeWidth="2"
              className="hover:opacity-80 transition-opacity cursor-pointer"
            />
            <text
              x={slice.labelX}
              y={slice.labelY}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="white"
              fontSize="12"
              fontWeight="bold"
              className="pointer-events-none"
            >
              {slice.percentage}%
            </text>
          </g>
        ))}
      </svg>
      
      <div className="flex flex-wrap justify-center gap-4 mt-4">
        {data.map((item, index) => (
          <div key={index} className="flex items-center space-x-2">
            <div 
              className="w-4 h-4 rounded"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-sm font-medium">{item.name}</span>
            <span className="text-sm text-muted-foreground">({item.value})</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export const SimpleBarChart: React.FC<SimpleBarChartProps> = ({ 
  data, 
  width = 500, 
  height = 300 
}) => {
  const margin = { top: 20, right: 30, bottom: 60, left: 40 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  
  const maxValue = Math.max(
    ...data.map(d => d.positive + d.negative + d.neutral)
  );
  
  const barWidth = chartWidth / data.length * 0.8;
  const barSpacing = chartWidth / data.length * 0.2;
  
  return (
    <div className="w-full">
      <svg width={width} height={height} className="overflow-visible">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Bars */}
          {data.map((item, index) => {
            const x = index * (barWidth + barSpacing) + barSpacing / 2;
            const totalHeight = (item.positive + item.negative + item.neutral) / maxValue * chartHeight;
            
            const positiveHeight = (item.positive / maxValue) * chartHeight;
            const negativeHeight = (item.negative / maxValue) * chartHeight;
            const neutralHeight = (item.neutral / maxValue) * chartHeight;
            
            return (
              <g key={index}>
                {/* Positive bar */}
                <rect
                  x={x}
                  y={chartHeight - positiveHeight}
                  width={barWidth}
                  height={positiveHeight}
                  fill="#22c55e"
                  className="hover:opacity-80 transition-opacity"
                />
                
                {/* Neutral bar */}
                <rect
                  x={x}
                  y={chartHeight - positiveHeight - neutralHeight}
                  width={barWidth}
                  height={neutralHeight}
                  fill="#6b7280"
                  className="hover:opacity-80 transition-opacity"
                />
                
                {/* Negative bar */}
                <rect
                  x={x}
                  y={chartHeight - totalHeight}
                  width={barWidth}
                  height={negativeHeight}
                  fill="#ef4444"
                  className="hover:opacity-80 transition-opacity"
                />
                
                {/* Category label */}
                <text
                  x={x + barWidth / 2}
                  y={chartHeight + 15}
                  textAnchor="middle"
                  fontSize="12"
                  fill="currentColor"
                  className="text-muted-foreground"
                >
                  {item.name.length > 10 ? item.name.substring(0, 10) + '...' : item.name}
                </text>
              </g>
            );
          })}
          
          {/* Y-axis */}
          <line
            x1={0}
            y1={0}
            x2={0}
            y2={chartHeight}
            stroke="currentColor"
            strokeWidth="1"
            className="text-border"
          />
          
          {/* X-axis */}
          <line
            x1={0}
            y1={chartHeight}
            x2={chartWidth}
            y2={chartHeight}
            stroke="currentColor"
            strokeWidth="1"
            className="text-border"
          />
        </g>
      </svg>
      
      {/* Legend */}
      <div className="flex justify-center gap-6 mt-4">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded bg-[#22c55e]" />
          <span className="text-sm">Positive</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded bg-[#6b7280]" />
          <span className="text-sm">Neutral</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded bg-[#ef4444]" />
          <span className="text-sm">Negative</span>
        </div>
      </div>
    </div>
  );
};