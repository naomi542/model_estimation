function p_hat = parzen1D(x, data, h)
N = size(data,1);
p_hat = zeros(size(x));

for i = 1:size(x,2)
    sum = 0;
    for j = 1:N
        sum = sum + normpdf(x(i), data(j), h);
    end
    p_hat(i) = sum / N;
end



