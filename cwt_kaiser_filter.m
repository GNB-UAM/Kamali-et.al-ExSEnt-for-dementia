% Continuous wavelet transfer signal with the Kaiser taper to extract the band-limited signal and power 

function [filtered_signal, phase_sig, band_power, freq_vec] = ...
         cwt_kaiser_filter(signal,Fs, f_low, f_high,  varargin)

% Adds:
%   phase_sig   – instantaneous phase (rad) via Hilbert on the filtered trace
%   band_power  – total power in [f_low f_high]  (same units as x^2)
%   A plot of |H(f)| and ∠H(f) whenever no output arguments are asked for.
%
% Optional flag remains:   'freq range','on'   → returns freq_vec = f(bandIdx)

% Parse optional flag 
showFreq = false;
if any(strcmpi(varargin, 'freq range'))          % quick test
    idx = find(strcmpi(varargin, 'freq range'),1);
    if numel(varargin) >= idx+1 && strcmpi(varargin{idx+1},'on')
        showFreq = true;
    end
end

%  CWT and taper 
                       
[wt, f] = cwt(signal,'amor',Fs);
bandIdx  = (f >= f_low) & (f <= f_high);
N_band   = nnz(bandIdx);

win  = kaiser(N_band, 5);    % smooth ( β = 3–8)
wt_mod(bandIdx,:) = wt(bandIdx,:) .* win;  

% Reconstruct  
sigMean = cast(mean(signal),'like',signal);   
filtered_signal = icwt(wt_mod,'amor','SignalMean',sigMean);


% Phase & power 
phase_sig  = unwrap(angle(hilbert(filtered_signal)));
band_power = abs(hilbert(filtered_signal)).^2; 

%  Optional outputs  
if showFreq
    freq_vec = f(bandIdx);
else
    freq_vec = [];
end

% Plot transfer function if no outputs
if nargout == 0
    H_mag          = zeros(size(f));   H_mag(bandIdx) = win;
    H_phase        = zeros(size(f));   H_phase(~bandIdx) = NaN;  % hide gaps

    figure('Name', 'CWT-Kaiser band-pass response');
    subplot(2,1,1)
        plot(f, H_mag, 'LineWidth',1.2);
        xlabel('Frequency (Hz)'), ylabel('|H(f)|'), grid on
        title('Magnitude response');xlim([0 40])
    subplot(2,1,2)
        plot(f, H_phase, 'LineWidth',1.2);
        xlabel('Frequency (Hz)'), ylabel('Phase (rad)'), grid on
        title('Phase response (always 0)');xlim([0 40])
end
end
