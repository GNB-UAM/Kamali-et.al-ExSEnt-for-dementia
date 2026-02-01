function hfd = Higuchi_FD(signal, kmax)
    % Default kmax if not provided
    if nargin < 2
        kmax = min(50, length(signal)-1); % Default to 50 or N-1
    end
    
    signal = signal(:);
    N = length(signal);
    
    % Handle constant signals (HFD=1 for straight line)
    if all(signal == signal(1))
        hfd = 1;
        return;
    end
    
    % Cap kmax to avoid invalid segments
    kmax = min(kmax, N-1);
    if kmax < 1
        hfd = NaN; % Insufficient data
        return;
    end
    
    L = zeros(1, kmax);
    
    for k = 1:kmax
        L_m = 0;
        valid_ms = 0; % Count valid m for current k
        
        for m = 1:k
            % Skip if subseries has <2 points
            if m > N - k
                continue;
            end
            
            valid_ms = valid_ms + 1;
            i_max = floor((N - m) / k); % Number of intervals
            
            % Sum absolute differences in subseries
            temp = 0;
            for i = 1:i_max
                idx1 = m + (i-1)*k;
                idx2 = m + i*k;
                temp = temp + abs(signal(idx2) - signal(idx1));
            end
            
            % Avoid division by zero (i_max>=1 always here)
            L_m = L_m + temp * (N-1) / (i_max * k^2);
        end
        
        % Average over valid m (if any)
        if valid_ms > 0
            L(k) = L_m / valid_ms;
        else
            L(k) = 0; % Fallback (unlikely due to kmax<=N-1)
        end
    end
    
    % Filter out L(k)=0 before regression
    valid_idx = (L > 0);
    if sum(valid_idx) < 2
        hfd = NaN; % Insufficient valid points
        return;
    end
    
    % Fit log(L) vs log(k)
    log_k = log(find(valid_idx)); % Use k-values where L>0
    log_L = log(L(valid_idx));
    p = polyfit(log_k, log_L, 1);
    hfd = -p(1); % HFD = -slope
end
