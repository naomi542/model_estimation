function ML = classify_ML(PA, PB, PC)
    ML = zeros(size(PA));
    for i= 1:size(PA,1)  % number of rows 
        for j=1:size(PA,2) % number of cols
            [max_value, class_index] = max([PA(i,j) PB(i,j) PC(i,j)]);
            ML(i,j) = class_index;
        end
    end 
end 