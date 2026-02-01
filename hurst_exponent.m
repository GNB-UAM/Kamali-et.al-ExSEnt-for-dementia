function H = hurst_exponent(data)
    if ~isvector(data)
        error('Input data must be a vector.');
    end
    
    data = data(:);
    N = length(data);
    
    % Handle constant data
    if all(data == data(1))
        H = NaN;
        return;
    end
    
    minWindow = 10;
    maxWindow = floor(N/2);
    
    % Handle insufficient data length
    if maxWindow < minWindow
        H = NaN;
        return;
    end
    
    windowSizes = round(logspace(log10(minWindow), log10(maxWindow), 20));
    windowSizes = unique(windowSizes);
    
    R_S_values = zeros(length(windowSizes), 1);
    
    for i = 1:length(windowSizes)
        w = windowSizes(i);
        numSegments = floor(N/w);
        RS_segment = zeros(numSegments, 1);
        
        for j = 1:numSegments
            segment = data((j-1)*w+1 : j*w);
            meanSegment = mean(segment);
            Y = cumsum(segment - meanSegment);
            R = max(Y) - min(Y);
            S = std(segment);
            
            if S > 0
                RS_segment(j) = R / S;
            else
                RS_segment(j) = 0;
            end
        end
        
        R_S_values(i) = mean(RS_segment);
    end
    
    % Check for sufficient valid R/S values
    positiveIndices = (R_S_values > 0);
    if sum(positiveIndices) < 2
        H = NaN;
        return;
    end
    logRS = log(R_S_values(positiveIndices));
    logWindowSizes = log(windowSizes(positiveIndices));
    
    p = polyfit(logWindowSizes, logRS, 1);
    H = p(1);
end
