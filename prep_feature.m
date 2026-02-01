%% Prepare the features tensor for the classification process 
clc; clear;

% --- Paths ---
data_path   = '/data path';
result_path = 'results path';
code_path   = 'codes path';
addpath(data_path, result_path, code_path);
cd(data_path)

% --- infodata ---
load('dementia_subject_cluster_info.mat');  % expects subjects_per_cluster, components_per_cluster

% --- Target clusters/bands ---
clusters      = [3,7,8,10];
cluster_names = {'RPFC','RVA','LVA','LPFC'};
for cl=1:numel(cluster_names)
    cls_name=cluster_names{cl};
    load(fullfile(result_path,sprintf('dementia_features_%s.mat',cls_name)))

    % --- Storage over clusters ---
    num_cls = numel(clusters);

    % Features' labels
    features_name = {'HD','HA','HDA','M','sampen','KFD','HFD','Hurst'};

    ntrls = 23;
    bands = band_names(2:end);
    export_dementia_tensors_pure_feats(feats, result_path, cls_name, features_name, ntrls, bands)

end
