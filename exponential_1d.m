function [lambda] = exponential_1d(data)
% ML estimate of mean is sample mean    
    mu = mean(data);
    lambda = inv(mu);
end