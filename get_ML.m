function [ML] = get_ML(X, Y, mean1, mean2, cov1, cov2)
    ML = zeros(size(X,1), size(Y,2));
    
    for i = 1:size(X, 1)
        for j = 1:size(Y, 2)
            point = [X(i,j) Y(i,j)];
            
            dist1 = get_MICD_distance(point, cov1, mean1);
            dist2 = get_MICD_distance(point, cov2, mean2);

            % if < 0, belongs to class 1; if > 0, belongs to class 2
            ML(i, j) = dist2 - dist1 - log(det(cov1)/det(cov2));
        end 
    end
end

function dist = get_MICD_distance(x, covariance, mean)
    dist = (x-mean)*inv(covariance)*(x-mean)';
end
