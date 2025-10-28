% Extract the band limited  power of EEG. We take the signals of each EEG channel of all the subjects, 
% apply a band-pass filter at 2 Hz to 48 Hz, remove drift and noise, apply cleanline 3 times at 50 Hz and 
% to remove line noise, apply the ASR to improve the bad segmenst and then common average reference 
% the data. Then we extract a 80 seconds data of each subject, from 120 to 200 second, to have balanced 
% length for all the subjects. We compute the  power using SFFT over 500 ms long windows with 50% overlap. 
% From this, for each epoch, we compute the mean power at each band, peak frequency at each band, 
% Hurst exponent and Sample Entropy at each band and also for the whole range of 3 to 60 Hz.
%This feature matrix is saved in a 4D data matrix to pass to DL models for classification!

data_path = 'C:\Users\SARA\Documents\MATLAB_files\data\Dementia';
eeglab_path = 'C:\Users\SARA\Documents\MATLAB_files\codes\eeglab2024.1';
code_path = 'C:\Users\SARA\Documents\MATLAB_files\codes\dementia';
addpath(data_path, eeglab_path, code_path)
eeglab;close

% Load Meta data (subject ID, gender, Age, disease group, MMSE score)
load('dementia_meta_data.mat');
num_sub = length(meta);

bands = {3:7.5, 8:12.5, 13:27.5, 28:48, 48:60};
n_bands = length(bands);
%% Load the data of all the subjects
for sub =1:num_sub
    sub_dir = sprintf('%s\\s%03d',data_path,sub);
    cd(sub_dir)
    file_name = sprintf('%s_task-eyesclosed_eeg.set', meta(sub).subject_id);
    EEG=pop_loadset(file_name,sub_dir);
    fs = EEG.srate;
    % High-pass filter
    f_low =2; % set it to 2 since for SFFT we set the window size to 500 ms
    f_high = 60;
    EEG = pop_eegfiltnew(EEG, 'locutoff', f_low, 'hicutoff',f_high ,'plotfreqz', 0);
    % Clean line noise
    for ii = 1:3
        EEG = pop_cleanline(EEG, 'bandwidth', 4, 'chanlist', 1:EEG.nbchan, ...
            'linefreqs', 50, 'newversion', 0, 'winsize', 7, 'winstep', 7);
    end

    % ASR-based cleaning
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 'off', 'ChannelCriterion', 'off', ...
        'LineNoiseCriterion', 'off', 'Highpass', 'off', 'BurstCriterion', 15,  ...
        'WindowCriterion', 0.2,'BurstRejection','on', 'Distance', 'Euclidian', 'WindowCriterionTolerances', [-Inf 7]);

    % Common average reference
    EEG = pop_reref( EEG, []);

    % Target frequency range (3 to 48 Hz with .5 Hz step)
    target_freqs = f_low:.5: f_high;

    segment_window_length = 2; % 1 seconds window for segmentation of data
    segment_window_length_idx = segment_window_length* fs; % Number of index per segment
    segments_overlap = 0; % no overlap
   
    % Extract a time window of the EEG to have balanced data of each subject
    Tstart=120; %start from after 120 seconds
    Tend = 280; % Extract 160 seconds of data
    start_idx=find(EEG.times>=Tstart*1000,1);
    end_idx = find(EEG.times>=Tend*1000,1);
    n_samples = end_idx-start_idx; % number of samples in the windowed data

    % Number of segmnets over selected time window
    n_segments = n_samples/segment_window_length_idx ;

    % We want to extract power at each band, the peak frequency at each band, and the peak
    % frequency in the whole range of frequencies lower than 28 and higher than 28 and the
    % hurst exponent and sample entropy of each band and the whole bands
   n_features = n_bands *5 + 5; 

   

    % Initiate matrix to save results
    features_matrix= zeros (EEG.nbchan, n_segments, n_features);
    for current_channel = 1: EEG.nbchan
        EEG_channel = EEG.data(current_channel,start_idx+1: end_idx);
        [s, f, t] = spectrogram(EEG_channel, segment_window_length_idx, segments_overlap, target_freqs, fs);
        % Power
        log_power =log10(abs(s).^2)'; % segments x frequencies
        power = (abs(s).^2)';

        % find the power, the peak frequency, Hurst exponet, sample entropy and Katz dimension at each band
        band_log_power = zeros(n_segments,n_bands);
        band_peak_fr = zeros(n_segments,n_bands);
        hurst_exp  = zeros(n_segments,n_bands);
        sam_en = zeros(n_segments,n_bands);
        m = 2; %embedding dimension
        r =0.2;% tolerance threshold
        KFD =  zeros(n_segments,n_bands);
        for band = 1:n_bands
            current_band= bands{band};
            idx1_band=find(target_freqs==current_band(1),1);
            idx2_band=find(target_freqs==current_band(end),1);
            band_log_power(:,band) = mean(log_power(:,idx1_band:idx2_band),2);
            [~,idx_max]= max(log_power(:,idx1_band:idx2_band),[],2);
            band_peak_fr(:,band) = target_freqs(idx_max+idx1_band-1);
            % Using arrayfun to apply hurst_exponent row by row
            hurst_exp(:, band) = arrayfun(@(row) hurst_exponent(power(row, idx1_band:idx2_band)), (1:size(log_power, 1))');
            % Using arrayfun to apply sample_entropy row by row
            sam_en(:, band) = arrayfun(@(row) sample_entropy(log_power(row, idx1_band:idx2_band), m, r), (1:size(log_power, 1))');
            % Using arrayfun to apply Katz_FD row by row
            KFD(:,band) = arrayfun(@(row) Katz_FD(power(row, idx1_band:idx2_band)), (1:size(log_power,1))');
        end
        fullband_log_power =  mean(log_power,2);
        [~,idx_max_glob]= max(normalize(log_power,2),[],2);
        dominant_peak_freq= target_freqs(idx_max_glob)';
        fullband_hurst_exp = arrayfun(@(row) hurst_exponent(power(row, :)), (1:size(log_power, 1))');
        fullband_samp_en =  arrayfun(@(row) sample_entropy(log_power(row, :), m, r), (1:size(log_power, 1))');
        fullband_KFD = arrayfun(@(row) Katz_FD(power(row, :)), (1:size(log_power,1))');
        features_matrix(current_channel, :, :) = ...
            [band_log_power,fullband_log_power, band_peak_fr, dominant_peak_freq, hurst_exp, ...
            fullband_hurst_exp, sam_en, fullband_samp_en, KFD, fullband_KFD];

    end

    % save the data of this subject
    file_name=sprintf('SFFT_19chan_features_s%03d.mat', sub);
    save(file_name, "features_matrix",'-v7.3');
    sprintf('Feature extraction for subject %d is done!', sub)
end
clearvars features_matrix
%% Loop through all subjects to create a 5D matrix for all the subjects
% Initiate 5D matrix to save the data of all the subjects
features_matrix4D = zeros(num_sub,EEG.nbchan, n_segments, n_features);

for sub = 1:num_sub
    sub_dir = sprintf('%s\\s%03d',data_path,sub);
    cd(sub_dir)
    file_name=sprintf('SFFT_19chan_features_s%03d.mat', sub);
     load(file_name);
    features_matrix4D(sub,:,:,:)= single(features_matrix);
end
cd(data_path)
% save the 4D data of power of all the subjects
save('dementia_features_data.mat','features_matrix4D','-v7.3')
