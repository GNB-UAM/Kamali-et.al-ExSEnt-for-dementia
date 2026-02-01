%% ++++ Preprocessing Dementia EEG & EEG Decomposition with AMICA  +++++
% clc; clear;

% Paths
data_path = 'data path';
eeglab_path = 'eeglab path';
code_path = 'codes path';
results_path ='path to save results';
addpath(data_path, eeglab_path, code_path,results_path)

% Load EEGLAB
eeglab; close;

% Load Meta data (subject ID, gender, Age, disease group, MMSE score)
load('dementia_meta_data.mat');
num_sub = length(meta);

good_subjects = [];
good_comps = {};
good_comps_counter = 0;
%% EEG Data Import and Preprocessing Loop

for current_subject =1:num_sub
    sub_dir = sprintf('%s\\s%03d',data_path,current_subject);
    cd(sub_dir)
    current_subject_id = meta(current_subject).subject_id;
    file_name = sprintf('%s_task-eyesclosed_eeg.set',current_subject_id);


    % Check if subject data file exists
    if ~exist(fullfile(sub_dir, file_name), 'file')
        fprintf('\n ---- WARNING: %s does not exist. Skipping this subject. ---- \n', subj_file_name);
        continue;
    end
    
    fprintf('\n\n\n---- Importing dataset for %s ----\n\n\n', file_name);
    
    % Load raw EEG data
    EEG=pop_loadset(file_name,sub_dir);
    fs = EEG.srate;
    % Make sure subjects' IDs are set consistently
    EEG.etc.initial_subject_id=EEG.subject;
    EEG.subject = [];
    EEG.subject = sprintf('%s%d',EEG.group,current_subject);
    
    % Band-pass filter
    f_low = 1;
    f_high = 100;
    EEG = pop_eegfiltnew(EEG, 'locutoff', f_low,'hicutoff', f_high, 'plotfreqz', 0);

    % Clean line noise, which are at 50 and 62.5 Hz
    for ii = 1:3
        EEG = pop_cleanline(EEG, 'bandwidth', 2,  'taperbandwidth', 1,'pad', 2,'chanlist', 1:EEG.nbchan, ...
         'linefreqs', [50, 62.5], 'p', 0.01, 'newversion', 0, 'normSpectrum',   0, 'winsize', 4, 'winstep', 2);
    end

   % Run ASR to clean the data, reconstruct the bad segments, and remove bad channels
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',4,'ChannelCriterion',0.6,...
        'LineNoiseCriterion',4,'Highpass','off','asrrej',0,'BurstRejection','off',...
        'BurstCriterion',15,'WindowCriterion','off','Distance','Euclidian',...
         'WindowLength',   0.5,'WindowCriterionTolerances',[-Inf 7] );

    % Average reference EEG channels
    EEG = pop_reref(EEG, []);

    % The nose is along the +X now. We want to rotate it to be on the +Y axis to align
    % with DIPFIT settings
    % define a 90° rotation about Z (in order to: X→ -Y & Y→X)
    theta = pi/2;     % 90° in radians
    Rz = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];

    for c = 1:numel(EEG.chanlocs)
        v = [EEG.chanlocs(c).X; EEG.chanlocs(c).Y; EEG.chanlocs(c).Z];
        w = Rz * v;
        EEG.chanlocs(c).X = w(1);
        EEG.chanlocs(c).Y = w(2);
        EEG.chanlocs(c).Z = w(3);
    end

    for ear_chan = 1:2
        v = [EEG.chaninfo.nodatchans(ear_chan).X; EEG.chaninfo.nodatchans(ear_chan).Y; ...
            EEG.chaninfo.nodatchans(ear_chan).Z];
        w = Rz * v;
        EEG.chaninfo.nodatchans(ear_chan).X = w(1);
        EEG.chaninfo.nodatchans(ear_chan).Y = w(2);
        EEG.chaninfo.nodatchans(ear_chan).Z = w(3);
    end
  
    % Create a new directory to save the IC results
    amica_output_dir='amica_out';
    mkdir(sub_dir,amica_output_dir);
    outdir_path=sprintf('%s\\%s',sub_dir,amica_output_dir);
    % ICA decomposition with AMICA
    EEG=pop_runamica(EEG,'num_mod',1,'pcakeep',EEG.nbchan -1,'outdir',outdir_path);
    
    % Source localization (MNI setting for dipfit)
    hdmfile=sprintf('%s\\plugins\\dipfit\\standard_BEM\\%s',eeglab_path,'standard_vol.mat');
    mrifile=sprintf('%s\\plugins\\dipfit\\standard_BEM\\%s',eeglab_path,'standard_mri.mat');
    chanfile=sprintf('%s\\plugins\\dipfit\\standard_BEM\\%s',eeglab_path,'elec\\standard_1020.elc');
    EEG =pop_dipfit_settings( EEG, 'hdmfile',hdmfile,'mrifile',mrifile,'chanfile',chanfile,...
        'coordformat','MNI','chansel', 1:EEG.nbchan );
    % Coregistr channels to fit the electrodes to the head model for dipol analysis
    [locs,EEG.dipfit.coord_transform] = coregister(EEG.chanlocs,EEG.dipfit.chanfile,...
        'chaninfo1', EEG.chaninfo,'mesh',EEG.dipfit.hdmfile,'warpmethod','rigidbody','warp','auto',...
        'manual','off');

    % Set the grid values for dipole fitting
    x_grid = -85:2:85;      %  from -85 to 85 
    y_grid = x_grid;             % Same for Y-axis
    z_grid = -1:2:85;         %  from 0 to 85

    % Run gridsearch dipole fitting, coarse grid search (residual‐variance cutoff = 0.4)
    EEG = pop_dipfit_gridsearch(EEG, [1:EEG.nbchan-1], x_grid, y_grid, z_grid, 0.4);

     % Save EEG in a temporary file
    cd(results_path);delete('tempEEG.set');delete('tempEEG.fdt');
    pop_saveset(EEG,'filename','tempEEG.set','savemode','twofiles', 'filepath',results_path);
    EEG=pop_loadset('tempEEG.set');

    % Run Talairach to find Brodmann area of each IC
    EEG=talLookup(EEG,[],'C:\Users\SARA\Documents\MATLAB_files\codes\');

    % Label ICs (running after fitting dipoles to use the dipoles inside the brain
    % metrics)
    EEG = pop_iclabel(EEG, 'default');

    % Find the components that are brain/eye components with rv<30%
    IC_label_probability=.60;dipole_residual=.3;
    EEG=dipfit_criteria(EEG,IC_label_probability,dipole_residual);

    % Update good subject list based on dipoles criteria
    if numel(EEG.components_to_cluster.brain_comps) > 0
        good_subjects = [good_subjects, current_subject];
        good_comps{end + 1} = EEG.components_to_cluster.good_comps;
        good_comps_counter = good_comps_counter + numel(EEG.components_to_cluster.good_comps);
    end

    % Import the data of subjects into the EEG
    EEG.etc.subjectInfo = meta(current_subject);

    % Setup event structure and epoch trials
    total_sample_points= size(EEG.data, 2);
    epoch_duration = 5; % desired epoch duration
    epoch_pnts = epoch_duration * fs; % sample points per 
    EEG.ntrial = floor(total_sample_points/epoch_pnts);
    EEG.event = event_struct_dementia(epoch_pnts, EEG.ntrial );
    EEG.pnts = epoch_pnts;
    EEG = pop_epoch( EEG, {}, [0  epoch_duration], 'epochinfo', 'yes');


    % Save final processed data
    pop_saveset(EEG, 'filename', sprintf('processed_sub%03d.set', current_subject), ...
        'savemode', 'twofiles', 'filepath', sub_dir);
    close %closing dipoles plot
end

% Save selected subjects and ICs for study design
cd(results_path);
save('good_subjects.mat', 'good_subjects');
save('good_comps.mat', 'good_comps');



