function [HD, HA, H_joint,M,range_D,range_A,r_D,r_A, segment_ids] = ExSEnt(signal,lambda,m,alpha)

% Inputs:
% signal: 1D time series
% lambda percentage of segments features IQR for noise tolerance
% m: embedding dimension
% alpha: noise tolerance for SampEnt
% Outputs:

    % Extract durations (D_vals) and cumulative amplitudes (A_vals)
    [D_vals, A_vals, segment_ids] = extract_DA(signal, lambda);
    
    % Check if valid segments were found
    if isempty(D_vals)
         error('No valid segments found in the raw signal with the current threshold.');
    end
    
    % Set parameters for sample entropy computation
    r_D = alpha * std(D_vals); % tolerance for durations (D)
    r_A = alpha * std(A_vals); % tolerance for amplitudes (A)
    
    % Compute sample entropy for durations (D)
    i=1;
    % for m=2:20
    HD(i) =  sample_entropy(D_vals, m, r_D);%sample_entropy(D_vals, m, r_D);
    
    % Compute sample entropy for cumulative amplitudes (A)
    HA(i) = sample_entropy(A_vals, m, r_A);%sampen


    % Compute oint sample entropy for the normalized paired (D, A)
    D_vals_norm=normalize(D_vals);  % Normalizes to [0,1]
    A_vals_norm=normalize(A_vals);  % Normalizes to [0,1]
    joint_data_norm = reshape([D_vals_norm(:) A_vals_norm(:)].',[],1);
    mAD = 2* m;
    
    % r_joint = alpha * std(joint_data(:)); % overall tolerance for joint data
    r_joint = alpha; % Tolerance for joint data since (std=1 for normalized data)
    H_joint(i) = sample_entropy(joint_data_norm, mAD, r_joint);
    % i=i+1;end
    % set(gca,'fontsize',12)
    % figure;subplot(311);plot(2:20,HD,'r-s','LineWidth',2);ylabel('H_D');hold on;subplot(312);plot(2:20,HA,'b-v','LineWidth',2);ylabel('H_A');
    % subplot(313);plot(2:20,H_joint,'k-d','LineWidth',2);ylabel('H_{DA}');xlabel('m');


    M=length(D_vals); %number of segments
    range_D = range(D_vals);%range of duration values
    range_A = range(A_vals);%range of amplitudes
    

end


