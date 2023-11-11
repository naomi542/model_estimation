function [class] = classify_points(X, Y, AB, BC, AC)
    class = zeros(size(AB));
    for i = 1:size(X, 1)
        for j = 1:size(Y, 2)
            class(i, j) = determine_class(AB(i,j), BC(i,j), AC(i,j));
        end
    end
end