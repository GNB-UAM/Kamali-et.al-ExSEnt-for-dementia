% Extract the band-limited signal and power of ICs at each cluster.
clc, clear all;

% Add paths
data_path = 'data path';
eeglab_path = 'eeglab path';
code_path = 'codes path';
results_path ='path to save results';
addpath(data_path, eeglab_path, code_path,results_path)

eeglab;close

% Load  data of the subjects and components at each cluster
load('dementia_subject_cluster_info.mat')

% Target clusters
clusters      = [3,7,8, 10];
cluster_names = {'RPFC', 'RVA','LVA','LPFC'};
num_cls = numel(clusters);

% Target frequency range (1 to 100 Hz with 0.5 Hz step)
frq_edges   = [ 1 4 ; 4 8 ; 8 12 ; 12  30 ; 30 48 ; 52 100 ];
band_names  = {'Delta','Theta','Alpha','Beta','Low Gamma','High Gamma'};
num_bands=numel(band_names);

% Initiate matrices to save results
dementia_bandpassed_signal_clusters=cell(1,num_cls);
min_trials=cell(1,num_cls);
num_trls_per_subject=cell(1,num_cls);
 
%% Loop over clusters to extract the bandpassed signals
for cl = 1:num_cls
    all_trls_num=[];
    cls = clusters(cl);
    mintrl=1000;
    subjects = subjects_per_cluster{cls};
    components = components_per_cluster{cls};
    num_sub = numel(subjects);
    all_subjects_pbsignal = cell(1,num_sub);
    for sub=1:num_sub
        current_subject = subjects(sub);
        sub_dir = sprintf('%s\\s%03d',data_path,current_subject);
        cd(sub_dir)
        file_name = sprintf('processed_sub%03d.set', current_subject);
        EEG=pop_loadset(file_name,sub_dir);
        % reshape continuous or epoched data: data = [nchans × (frames×trials)]
        fs=EEG.srate;
        data = reshape( EEG.data, EEG.nbchan, [] );
        mintrl = min(mintrl,EEG.ntrial);
        % compute all activations: [ncomps × (frames×trials)]
        icaActAll = ( EEG.icaweights * EEG.icasphere ) * data;
        % optionally reshape back to [ncomps × frames × trials]
        if EEG.trials > 1
            icaActAll = reshape(icaActAll, EEG.nbchan-1, EEG.pnts, EEG.trials);
        end
        % assign to EEG.icaact
        EEG.icaact = icaActAll;
        current_comp = components(sub);
        signal_bp = zeros(size(icaActAll,2)*EEG.ntrial,num_bands);
        phase_bp= zeros(size(icaActAll,2)*EEG.ntrial,num_bands);
        power_bp= zeros(size(icaActAll,2)*EEG.ntrial,num_bands);
        frq_bands = cell(1,num_bands);
        for band=1:num_bands
            f_low = frq_edges(band,1);
            f_high = frq_edges(band,2);
           signal = EEG.icaact(current_comp,:,:);
           signal = signal(:);

           [signal_bp(:,band), phase_bp(:,band), power_bp(:,band) , ~] = ...
               cwt_kaiser_filter(signal, fs, f_low, f_high);
           %    [filtered_signal(:,band), band_power(:,band) , band_phase(:,band), ~] = ...
           % butter_bandpass_hilbert(signal, fs, f_low, f_high, order);

        end
        file_name = sprintf('bandpassed_s%03d_IC%d.mat',current_subject,current_comp);
        save(file_name,"signal_bp")
        file_name = sprintf('band_phase_s%03d_IC%d.mat',current_subject,current_comp);
        save(file_name,"phase_bp")
        file_name = sprintf('band_power_s%03d_IC%d.mat',current_subject,current_comp);
        save(file_name,"power_bp")
        all_subjects_pbsignal{sub}=signal_bp;
        all_trls_num = [all_trls_num, EEG.ntrial];
        sprintf('%.2f of subjects of cluster %d: done!', sub/num_sub,cls)
    end
    dementia_bandpassed_signal_clusters{cls}=all_subjects_pbsignal;
    min_trials{cls}=mintrl;
    num_trls_per_subject{cls} = all_trls_num;
    sprintf('%.2f of clusters done!',cl/num_cls)
end
cd(result_path)
save('dementia_subject_cluster_info.mat',"subjects_id",...
    "components_per_cluster","subjects_per_cluster",'stats','uniq_subjects','min_trials')
save('dementia_bpsignal_mega_data.mat','dementia_bandpassed_signal_clusters',...
    "num_trls_per_subject", '-v7.3')
