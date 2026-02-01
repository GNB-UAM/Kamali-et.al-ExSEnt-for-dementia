% This code generates a study file for dementia data analysis based on previously processed and cleaned EEG data,
% for subjects with at least one brain dipole. It only keeps the brain and eye components and removes the rest from the analysis.
clc; clear all;

% Set paths for EEGLAB, codes, and subject data directories
data_path = 'data path';
eeglab_path = 'eeglab path';
code_path = 'codes path';
results_path ='path to save results';
addpath(data_path, eeglab_path, code_path,results_path)


% Initialize EEGLAB and add required paths
addpath(eeglab_path); eeglab; close;

load("good_subjects.mat"); %WE already saved the good subjects with the good ICs when we preprocessed the data
subjects = good_subjects;

% Create study
nsubj= length(subjects);
% Preallocate cell array to store good components for each subject
components = cell(1, nsubj);

% Initialize ALLEEG structure
ALLEEG = struct();

% Loop over subjects to load data and extract good components
for current_subj = 1:nsubj
    subj = subjects(current_subj);
    eeg_file_out_dir = fullfile(data_path,sprintf('s%03d', subj));
    cd(eeg_file_out_dir);  % Change directory to the current subject's file directory

    % Load the processed EEG data file
    EEG_file_name = sprintf('processed_sub%03d.set', subj);
    EEG = pop_loadset('filename', EEG_file_name, 'filepath', eeg_file_out_dir);

    % Store good components for clustering
    components{current_subj} = EEG.components_to_cluster.good_comps;

    % Store EEG data in ALLEEG structure
    [ALLEEG, ~, ~] = eeg_store(ALLEEG, EEG, current_subj);
end


% Create and configure the STUDY structure
STUDY = struct();
[STUDY, ALLEEG] = std_editset(STUDY, ALLEEG, 'name', 'goodSubjects', 'updatedat', 'on', ...
    'task', 'closed_eyes', 'filename', 'dementia.study', 'filepath', results_path);

% Add only good components to the STUDY structure
for sub = 1:nsubj
    [STUDY, ALLEEG] = std_editset(STUDY, ALLEEG, 'commands', ...
        {{'index' sub 'comps' components{sub}}}, 'filename', 'dementia_initial.study', ...
        'filepath', results_path);
end

%% Precompute the power spectrum and scalp map and cluster the ICs
[STUDY, ALLEEG] = pop_loadstudy('filename', 'dementia_initial.study', 'filepath', results_path);
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'components','savetrials','on','recompute','on',...
    'scalp','on','spec','on','specparams',{'freqrange' [1 100]});

% Cluster the study by defining clustering parameters' weights, and cluster the dipoles with the k-means method

[STUDY, ALLEEG] = std_preclust(STUDY, ALLEEG, [], {'dipoles', 'weight', 10},  {'spec', 'weight', 0}, ...
    {'scalp', 'weight', 1}); 
    % spe's weight 0, since power is used with RL for classification (avoid double-dipping)

number_of_clusters = 10; %This is based on some search with the GUI to find the stable number 
[STUDY] = pop_clust(STUDY, ALLEEG, 'algorithm', 'kmeans', ...
    'clus_num',  number_of_clusters);

[STUDY, ALLEEG] = pop_savestudy(STUDY, ALLEEG, 'filename', 'dementia_clustered.study', ...
    'filepath', results_path); 


%% -------------------------------------------------------------------------
%  FILTER CLUSTER MEMBERS BY SPECTRUM- & SCALPMAP-SIMILARITY, ONE IC / SUBJECT
%Load the study

[STUDY, ALLEEG] = pop_loadstudy('filename', 'dementia_clustered.study', 'filepath', results_path);

num_cls        = numel(STUDY.cluster);

for cl = 2:num_cls    % skip parent, cl=1, “All components”
    % -----------subjects and components data for this cluster --------------
    cls_subjects = STUDY.cluster(cl).sets;
    cls_comps = STUDY.cluster(cl).comps;

    % ----------- gather spectra & topo maps for this cluster --------------
    grid=cell(1, numel(cls_comps) );
    for k = 1:numel(cls_comps)
        [grid{k}, ~, ~] = std_readtopo(ALLEEG, cls_subjects(k), cls_comps(k));
    end
    [STUDY,spec, ~] =  std_readspec(STUDY, ALLEEG, 'datatype', 'spec', 'clusters',cl);

    % --- 1. stack data ------------------------------------------------------
    nComp   = numel(grid);                 % number of ICs in the cluster
    topo3d  = cat(3, grid{:});             % [row × col × comp]
    meanTopo = mean(topo3d, 3, 'omitnan');           % grand-average scalp map

    specMat = cat(2, spec{:}); %stack the spectrum data of all the groups to apply for loop
    % compute the mean spectrum for each group separately
    meanSpec = struct('A',mean(spec{1}, 2), 'C', mean(spec{2}, 2), 'F', mean(spec{3}, 2));

    % Cleaning ICs based on topo and spectrum similarity 
    topo_sim = zeros(nComp,1);
    spec_qual = zeros(nComp,1);
    tA=[];tB=[];nan_mask_topo=[];
    sA = []; sB=[];
    for ii = 1:nComp
        current_topo = topo3d(:,:,ii);
        % Topography similarity
        tA   = current_topo(:);
        tB   = meanTopo(:);
        nan_mask_topo   = ~isnan(tA) & ~isnan(tB);      % pairwise-valid mask
        tA   = tA(nan_mask_topo);  tB = tB(nan_mask_topo);
        topo_sim(ii) = abs(corr(tA, tB));          
    end

    % Set threshold for matching group mean
    topo_corr_thr = 0.55;   % scalp-map   similarity

    keep  = (topo_sim >= topo_corr_thr);% & (spec_qual);

    cls_comps   = cls_comps(keep);          % filtered component indices
    cls_subjects = cls_subjects(keep);       % matching subject indices
    topo_sim = topo_sim(keep);
    % spec_qual = spec_qual(keep);

    % Inspect results --------------------------------------
    % grid  = grid(keep);               % kept topographies
    % specMat  = specMat(:,keep);               % kept spectra
    % simTable = table((1:nComp)', topoSim, specSim, keep, ...
    %     'VariableNames',{'Idx','Topo_r','Spec_r','Kept'});
    % disp(simTable);                          % quick sanity check

    subjects  = cls_subjects(:);                  % column vector
    comps     = cls_comps(:);                     % matching component numbers

    uniqSub   = unique(subjects);                 % list of subjects
    keep_one  = false(numel(subjects),1);         % final mask

    for s = 1:numel(uniqSub)
        current_uniq_sub = uniqSub(s);
        idx      = find(subjects==current_uniq_sub);     % all ICs of this subject
        [bestVal, ~] = max(topo_sim(idx));                  % highest similarity
        cand      = idx( topo_sim(idx) >= bestVal );        % potential ties
        if numel(cand) > 1                                 % tie → smallest comp#
            [~,m]  = min(comps(cand));
            cand   = cand(m);
        end
        keep_one(cand) = true;         % flag the winner
    end

    % Apply mask and  update the cluster inside STUDY --------------------------------
    STUDY.cluster(cl).sets  =  cls_subjects(keep_one);        % dataset indices
    STUDY.cluster(cl).comps = cls_comps(keep_one);       % component numbers
    % Let EEGLAB recompute internal fields
    STUDY = std_checkset(STUDY, ALLEEG);      % refresh the STUDY structure

end

% Save the STUDY ----------
[STUDY, ALLEEG] = pop_savestudy(STUDY, ALLEEG, 'filename', 'dementia_cleaned.study', ...
    'filepath', results_path);

% Extract the information of the ICs and Subjects in each cluster
%Load study
[STUDY, ALLEEG] = pop_loadstudy('filename', 'dementia_cleaned.study', 'filepath', results_path);
num_cls        = numel(STUDY.cluster);
subjects_per_cluster = cell(1,num_cls);
components_per_cluster = cell(1,num_cls);
subjects_id= cell(1,num_cls);
stats= cell(1,num_cls);
all_subjects=[];
for cls = 2:num_cls                               % skip parent “All components”
    % -----------subjects and components data for this cluster --------------
    subjects_per_cluster{cls} = STUDY.cluster(cls).sets;
    components_per_cluster{cls} = STUDY.cluster(cls).comps;
    ids=cell(1,numel(components_per_cluster{cls} ));
    grp=cell(1,numel(components_per_cluster{cls} ));
    cls_subjects=STUDY.cluster(cls).sets;
    for k = 1:numel(components_per_cluster{cls} )
        ids{k}= STUDY.datasetinfo(cls_subjects(k)).subject;
        grp{k}= STUDY.datasetinfo(cls_subjects(k)).group;
    end

    symbols  = {'A' 'C' 'F'};
    % count each symbol
    nPerSym  = cellfun(@(s) sum(strcmp(grp,s)), symbols);
    % build structure
    stats{cls} = cell2struct( num2cell(nPerSym) , symbols , 2 );

    subjects_id{cls} =ids;
    all_subjects=[all_subjects,subjects_per_cluster{cls}];
end
uniq_subjects=unique(all_subjects);
cd(results_path)
save('dementia_subject_cluster_info.mat',"subjects_id","components_per_cluster",...
    "subjects_per_cluster",'stats','uniq_subjects')
