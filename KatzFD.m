%This function computes the Katz fractal dimension of the input, which is a one
%dimensional time-series. Where data,  is the amplitude of the signal.
%N, is the total number of data points in the data.

function [KFD,max_dist,eu_length]= KatzFD(data)
if ~isnan(data)
    N = length(data);Euclidian_dist= zeros(1,N-1);
    for i = 1:N-1
        %Euclidian distance between consecutive data points:
        Euclidian_dist(i) = sqrt((data(i+1)-data(i))^2);
    end
    eu_length = sum(Euclidian_dist);%adding them to get the total Euclidian length of data


    %Computing the distances between first data point and all the other data points:
    pair_dist = zeros(N,N-1);
    for i = 1:N-1
        for j=i+1:N
            pair_dist(i,j) = sqrt((data(i)-data(j))^2);
        end
    end
    max_dist = max(pair_dist(:));%maximum distance between the first point and any other point of data
    % Computing KFD:
    KFD = log10(N-1)/(log10(N-1) + log10(max_dist/eu_length));
else
    KFD=NaN;
    warning('The input signal is Nan!')
end
