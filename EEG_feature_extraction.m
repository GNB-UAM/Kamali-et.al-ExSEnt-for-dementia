% Extract the band limited  power of EEG. We take the signals of each EEG channel of all the subjects, 
% apply a band-pass filter at 2 Hz to 48 Hz, remove drift and noise, apply cleanline 3 times at 50 Hz and 
% to remove line noise, apply the ASR to improve the bad segmenst and then common average reference 
% the data. Then we extract a 80 seconds data of each subject, from 120 to 200 second, to have balanced 
% length for all the subjects. We compute the  power using SFFT over 500 ms long windows with 50% overlap. 
% From this, for each epoch, we compute the mean power at each band, peak frequency at each band, 
% Hurst exponent and Sample Entropy at each band and also for the whole range of 3 to 60 Hz.
%This feature matrix is saved in a 4D data matrix to pass to DL models for classification!

data_path = 'path to data';
eeglab_path = 'path to eeglab';
code_path = 'path to codes root';
addpath(data_path, eeglab_path, code_path)
eeglab;close

% Load subjects info file (subject ID, gender, Age, disease group, MMSE score)
load('dementia_meta_data.mat');
num_sub = length(meta);

clusters      = [3,7,8,10];
cluster_names = {'RPFC','RVA','LVA','LPFC'};


frq_edges     = [ 1 4 ; 4 8 ; 8 12 ; 12 30 ; 30 48 ; 52 100 ];
band_names    = {'Delta','Theta','Alpha','Beta','Low Gamma','High Gamma'};
selected_bands = [2,3,4,5,6];
num_bands     = numel(band_names);
num_selected_bands     = numel(selected_bands);

fs   = 500;

% --- Feature parameters ---
m        = 2;       % embedding dimension
lambda   = 0.01;
alpha    = 0.2;     % tolerance (Sample Entropy)
kmax     = 20;      % for Higuchi FD
trial_duration = 24;     % The length of the windows to compute featires: 24 seconds
win_len     = trial_duration * fs;
overlap = 0.5; % 50% overlap 
step =round(win_len * (1 - overlap)) ; %step size in computing features

% --- Storage over clusters ---
num_cls = numel(clusters);
all_features_clusters   = cell(1, num_cls);
num_trls_per_subject    = cell(1, num_cls);

%% Loop over clusters
for icl = 1:num_cls
    cls        = clusters(icl);
    subjects   = subjects_per_cluster{cls};
    components = components_per_cluster{cls};
    num_sub    = numel(subjects);
    cluster_features = cell(1, num_sub);  % one entry per subject

    for isub = 1:num_sub
        current_subject = subjects(isub);
        current_comp    = components(isub);
        % --- Load bandpassed signal, phase, power ---
        sub_dir = sprintf('%s/s%03d', data_path, current_subject);
        cd(sub_dir);

        load(sprintf('bandpassed_s%03d_IC%d.mat', current_subject, current_comp), 'signal_bp');
        load(sprintf('band_power_s%03d_IC%d.mat',   current_subject, current_comp), 'power_bp');

        % --- Discard edges: 3.5 s each side (7-cycle wavelet at fs=500) ---
        cut = 1750;  % samples to remove to discard filter edge effect
        signal_bp = signal_bp(cut+1:end-cut, :);   % [T x num_bands]
        phase_bp  = phase_bp (cut+1:end-cut, :);
        power_bp  = power_bp (cut+1:end-cut, :);

        % --- Epoching ---
        N = size(signal_bp, 1);
        nwin = 23; % We only want the features over the first 290 seconds. Which will generate 57 overlapping windows
        ntrials = 12;

        % Trim data
        signal_bp = signal_bp(1:ntrials*win_len, :);
        phase_bp  = phase_bp (1:ntrials*win_len, :);
        power_bp  = power_bp (1:ntrials*win_len, :);

        % --- Preallocate per-subject feature arrays ---
        HD         = nan(nwin, num_selected_bands);
        HA         = nan(nwin, num_selected_bands);
        HDA    = nan(nwin, num_selected_bands);
        M      = nan(nwin, num_selected_bands);
        sampen     = nan(nwin, num_selected_bands);
        KFD        = nan(nwin, num_selected_bands);
        HFD        = nan(nwin, num_selected_bands);
        Hurst_exp  = nan(nwin, num_selected_bands);
        power_mean = nan(nwin, num_selected_bands);


        % --- Trial loop ---
        for ww = 1:nwin
           start_idx = (ww-1) * step + 1;
           end_idx   = start_idx + win_len - 1;
           signal = signal_bp(start_idx:end_idx,:);
           power = power_bp(start_idx:end_idx,:);


            % --- Per-band features ---
            for b= 1:numel(selected_bands)
                band = selected_bands(b);
                x = squeeze(signal(:, band));
                [HD(ww,b), HA(ww,b), HDA(ww,b), M(ww,b), ...
                    ~, ~, ~, ~, ~] = ExSEnt(x, lambda, m, alpha);
                r = alpha * std(x);
                sampen(ww,b) = sample_entropy(x, m, r);
                KFD(ww,b)       = KatzFD(x);
                HFD(ww,b)       = Higuchi_FD(x, kmax);
                Hurst_exp(ww,b) = hurst_exponent(x);
                power_mean(ww,b)= mean(power(:, band));
            end
        end

        % --- Pack features for this subject ---
        features = struct();
        features.info.subject     = current_subject;
        features.info.ic          = current_comp;
        features.info.cluster_id  = cls;
        features.info.cluster_name= cluster_names{icl};
        features.info.fs          = fs;
        features.info.band_names  = band_names;
        features.info.selected_bands = selected_bands;
        features.info.trial_pnts  = win_len;
        features.info.trial_secs  = trial_duration;

        features.HD = HD;
        features.HA = HA;
        features.HDA = HDA;
        features.M = M;
        features.sampen = sampen;
        features.KFD = KFD;
        features.HFD = HFD;
        features.Hurst = Hurst_exp;
        features.powermean = power_mean;

        cluster_features{isub}  = features;
        num_trls_per_subject{icl}(isub) = nwin;

        fprintf('Cluster %d (%s): %.02f%% of subjects done!\n', ...
            icl, cluster_names{icl}, isub/num_sub*100);
        cd(sub_dir);
        save(sprintf('features_sub%03d_ic%d_5bands.mat',isub,icl),"features")
    end

    all_features_clusters{icl} = cluster_features;
   
feats =all_features_clusters{icl};
    cd(result_path);
save(sprintf('dementia_features_%s.mat',cluster_names{icl}), ...
    'feats', 'num_trls_per_subject', ...
    'clusters', 'cluster_names', 'band_names', ...
    'phase_bands_idx', 'amp_bands_idx', 'nbins_pac', '-v7.3');

 fprintf('%.2f of clusters done and saved!\n', icl/num_cls);  %Print the progress!

end

% --- Save ---
cd(result_path);
save('dementia_features_5bands.mat', ...
    'all_features_clusters', 'num_trls_per_subject', ...
    'clusters', 'cluster_names', 'band_names', ...
    'phase_bands_idx', 'amp_bands_idx', 'nbins_pac', '-v7.3');

fprintf('Saved: dementia_features.mat\n');

