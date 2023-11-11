% 
% Gaussian Parameter Estimation - estimate mu and sigma using ML
% tbh im taking these equations straight from slide 8 slides sooo idk if
% we're supposed to show more lol
%
% [mu, sigma] = gaussian_param_est(data)
% 
% data - one or two column matrix of points
% 
% mu - mean vector of gaussian
% sigma - standard deviation or covariance matrix of gaussian

function [mu,sigma] = gaussian_param_est(data)
% ML estimate of mean is sample mean    
mu = mean(data);

% ML estimate of std dev (and variance) is just sample std dev (var)
num_samples = size(data,1);
sum = 0;
for i = 1:num_samples
    sum = sum + (data(i,:) - mu)'*(data(i,:) - mu);
end
sigma = sum / num_samples;

    
    