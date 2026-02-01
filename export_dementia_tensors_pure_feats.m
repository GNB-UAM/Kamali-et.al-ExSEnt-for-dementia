function export_dementia_tensors_pure_feats(data, out_dir, cluster_name, feat_list, ntrls, band_names, varargin)
% export_dementia_tensors_pure_feats
% Build per-cluster TRIAL-WISE tensors of "pure" metrics (no ratios).
% Ordinary metrics: over all bands in band_names (e.g., 5 bands).
% Special metrics: MI_Tort and PrefPhase → only (phase=alpha, amp={lowgamma, highgamma}).
%
% X shape: [N_subj, ntrls, nfeat]
%   nfeat = (#ordinary_metrics * nbands) + (2 if MI in feat_list) + (2 if PrefPhase in feat_list)
%
% NaNs are kept (impute later).
%
% Sara Kamali 2025-08-13

if nargin < 2 || isempty(out_dir), out_dir = fullfile(pwd, 'exports'); end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
if nargin < 3 || isempty(cluster_name), cluster_name = 'cluster'; end
if nargin < 4 || isempty(feat_list), error('feat_list is required'); end
if nargin < 5 || isempty(ntrls), error('ntrls is required'); end
if nargin < 6 || isempty(band_names), error('band_names is required'); end

nbands = numel(band_names);


n_ordinary = sum(is_ordinary);
nfeat = n_ordinary * nbands + 2*any(is_MI) + 2*any(is_Pref);

% --- Gather usable subjects
N = numel(data);
usable_idx = false(N,1);
for s = 1:N
    S = get_subject_struct(data{s});
    if ~isempty(S), usable_idx(s) = true; end
end
idx = find(usable_idx);
Nsubj = numel(idx);


% --- Preallocate
XX = nan(Nsubj, ntrls, nfeat);
ymmse  = nan(Nsubj, 1);
subject_id     = cell(Nsubj,1);
group_str      = cell(Nsubj,1);
group_numeric  = nan(Nsubj,1);
age            = nan(Nsubj,1);
gender_raw     = cell(Nsubj,1);
gender_numeric = nan(Nsubj,1);

% --- Build feature_names (same order we will fill XX)
feature_names = {};
% Metrics over all bands
for f = find(is_ordinary)
    for b = 1:nbands
        feature_names{end+1} = sprintf('%s_%s', feat_list{f}, band_names{b}); %#ok<AGROW>
    end
end

% --- Fill XX
k = 0;
for s = idx(:).'
    S = get_subject_struct(data{s});
    if isempty(S), continue; end
    k = k + 1;

    % Build subject feature matrix in the same column order used above
    cols = [];

    % Ordinary metrics → each as [ntrls × nbands]
    for f = find(is_ordinary)
        fname = feat_list{f};
        Fi = get_field_from_struct(S, fname);
        Fi = ensure_2d(Fi);
        Fi = pad_trunc_2D(Fi, ntrls, nbands);  % ntrls×nbands
        cols = [cols, Fi]; %#ok<AGROW>
    end


    if ~isequal(size(cols), [ntrls, nfeat])
        error('Subject %d produced %s matrix, expected [%d %d].', s, mat2str(size(cols)), ntrls, nfeat);
    end
    XX(k,:,:) = cols;

    % Labels / meta
    ymmse(k) = get_field_from_struct(S, 'MMSE');
    sid = get_field_from_struct(S, {'subject_id','subID','ID','id','subject'});
    subject_id{k} = to_str(sid);
    grp = get_field_from_struct(S, {'group','Group'});
    group_str{k} = to_str(grp);
    group_numeric(k) = map_group_label(group_str{k});  % A=1, F=2, C=0
    ag = get_field_from_struct(S, {'age','Age'});
    if ~isempty(ag), age(k) = double(ag); end
    gdr = get_field_from_struct(S, {'gender','Gender','sex'});
    gender_raw{k} = to_str(gdr);
    gender_numeric(k) = map_gender(gender_raw{k});     % F=0, M=1, else NaN
end

% --- Verify shapes
assert(all(size(XX)==[Nsubj,ntrls,nfeat]), 'X shape must be Nsubj×ntrls×nfeat.');
assert(isvector(ymmse) && numel(ymmse)==Nsubj, 'y must be Nsubj×1.');

% --- Label maps
label_map.group.A = 1;
label_map.group.F = 2;
label_map.group.C = 0;
label_map.gender.F = 0;
label_map.gender.M = 1;

[yCD, yAF, yCE, yEL, detail] = make_labels_mmse(ymmse, group_str);

% --- Save MAT (v7 so scipy can load easily)
matfile_name = fullfile(out_dir, sprintf('tensor_pure_feats_%s.mat', cluster_name));
save(matfile_name, 'XX','ymmse','yCD','subject_id','group_str','group_numeric', ...
    'age','gender_raw','gender_numeric','feature_names','band_names', ...
    'cluster_name','label_map',"yAF","yCE","yEL",'detail','-v7');

fprintf('[%s] Saved %d subjects → %s\n', cluster_name, Nsubj, matfile_name);
end

% ===== Helpers =====
function S = get_subject_struct(node)
    S = [];
    if isempty(node), return; end
    if isstruct(node), S = node; return; end
    if iscell(node)
        for i=1:numel(node)
            if isstruct(node{i}), S = node{i}; return; end
        end
    end
end

function val = get_field_from_struct(S, names)
    if ischar(names) || (isstring(names) && isscalar(names)), names = {char(names)}; end
    val = [];
    for i = 1:numel(names)
        nm = names{i};

        % PAC special-cases (pass through for ordinary lookup too)
        if strcmpi(nm,'MI_Tort')
            if isfield(S,'PAC') && isfield(S.PAC,'MI_Tort'), val = S.PAC.MI_Tort; return; end
            if isfield(S,'all_features') && isfield(S.all_features,'PAC') && isfield(S.all_features.PAC,'MI_Tort')
                val = S.all_features.PAC.MI_Tort; return;
            end
        end
        if any(strcmpi(nm,{'PrefPhase','PreferredPhase','Pref_Phase'}))
            if isfield(S,'PAC') && (isfield(S.PAC,'PrefPhase') || isfield(S.PAC,'PreferredPhase'))
                if isfield(S.PAC,'PrefPhase'), val = S.PAC.PrefPhase; else, val = S.PAC.PreferredPhase; end
                return;
            end
            if isfield(S,'all_features') && isfield(S.all_features,'PAC')
                P = S.all_features.PAC;
                if isfield(P,'PrefPhase'), val = P.PrefPhase; return; end
                if isfield(P,'PreferredPhase'), val = P.PreferredPhase; return; end
            end
        end

        % generic lookups
        if isfield(S,nm), val = S.(nm); return; end
        if isfield(S,'features') && isfield(S.features,nm), val = S.features.(nm); return; end
        if isfield(S,'all_features') && isfield(S.all_features,nm), val = S.all_features.(nm); return; end
    end
end

function A = ensure_2d(A)
    if isempty(A), return; end
    A = double(A);
    if isvector(A)
        A = A(:); % column
    end
    if ndims(A) > 2
        % squeeze if e.g. T×B×1
        A = squeeze(A);
        if ~ismatrix(A)
            error('Unexpected array shape for feature (ndims>2).');
        end
    end
end

function M = pad_trunc_2D(A, T, B)
    M = nan(T, B);
    if isempty(A), return; end
    A = double(A);
    t = min(T, size(A,1));
    b = min(B, size(A,2));
    M(1:t, 1:b) = A(1:t, 1:b);
end

function M2 = pad_trunc_2D_2cols(A, T)
    % Return T×2 [col1 col2], NaN-padded/truncated
    M2 = nan(T, 2);
    if isempty(A), return; end
    A = ensure_2d(A);
    t = min(T, size(A,1));
    c = min(2, size(A,2));
    M2(1:t, 1:c) = A(1:t, 1:c);
end

function out = to_str(x)
    if isstring(x) || ischar(x), out = char(x); return; end
    if iscellstr(x) && numel(x)==1, out = x{1}; return; end
    if isnumeric(x) && isscalar(x), out = sprintf('%g', x); return; end
    if isempty(x), out = ''; return; end
    try, out = char(string(x)); catch, out = ''; end
end

function g = map_group_label(s)
    g = nan; if isempty(s), return; end
    if any(strcmp(s, {'C'})), g = 0; return; end
    if any(strcmp(s, {'A'})), g = 1; return; end
    if any(strcmp(s, {'F'})), g = 2; return; end
end

function gn = map_gender(gstr)
    gn = nan; if isempty(gstr), return; end
    s = lower(strtrim(gstr));
    if any(strcmp(s, {'f','female'})), gn = 0; return; end
    if any(strcmp(s, {'m','male'})),   gn = 1; return; end
end


