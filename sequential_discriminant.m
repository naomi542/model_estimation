function [discriminant] = sequential_discriminant(a, b, J, classify_points)
    if nargin < 3 
        J = inf; 
        classify_points = false;
    end;
    
    full_a = a;
    full_b = b;

    increment = 1;
    x1 = min([a(:,1);b(:,1)]):increment:max([a(:,1);b(:,1)]);
    x2 = min([a(:,2);b(:,2)]):increment:max([a(:,2);b(:,2)]);
    [X1, X2] = meshgrid(x1,x2);
    
    classifiers = zeros(size(X1,1),size(X2,2),1);
    n_aB = [];
    n_bA = [];
    
    rand_points_A = zeros(1,2);
    rand_points_B = zeros(1,2);

    while((size(a,1) ~= 0 || size(b,1) ~= 0) && size(classifiers, 3)-1 ~= J)
        pointA = a(randi(size(a,1)), :);
        pointB = b(randi(size(b,1)), :);
        
        G = get_MED(X1, X2, pointA, pointB);
        [A, B] = classification_error(a, b, pointA, pointB);
        n_aB_j = size(A,1);
        n_bA_j = size(B,1);

        if(n_aB_j == 0 || n_bA_j == 0)
            % Discriminant is good, save G
            classifiers_copy = classifiers;
            classifiers = zeros(size(classifiers,1),size(classifiers,2),size(classifiers,3)+1);
            classifiers(:,:,1:size(classifiers,3)-1) = classifiers_copy;
            classifiers(:,:,size(classifiers,3)) = G;
            
            n_aB = [n_aB; n_aB_j];
            n_bA = [n_bA; n_bA_j];
            
            % Remove correctly classified points
            if(n_aB_j == 0) b = B; end
            if(n_bA_j == 0) a = A; end
            
            rand_points_A = [rand_points_A; pointA];
            rand_points_B = [rand_points_B; pointB];
        end
    end
    
    if classify_points
        discriminant = sequential_classify(full_a, full_b, rand_points_A, rand_points_B, n_aB, n_bA);  
    else
        discriminant = create_discriminant(X1, X2, classifiers, n_aB, n_bA); 
    end
end

function error = sequential_classify(a, b, rand_points_A, rand_points_B, n_aB, n_bA)
    error = 0;
    
    for i = 1:size(a,1)
        for j = 1:size(n_aB,1) 
            if(get_distance(a(i,:), rand_points_A(j+1,:)) - get_distance(a(i,:), rand_points_B(j+1,:)) < 0 && n_bA(j) == 0)
                break;
            elseif(get_distance(a(i,:), rand_points_A(j+1,:)) - get_distance(a(i,:), rand_points_B(j+1,:)) > 0 && (n_aB(j) == 0) || (j == size(n_aB,1)))
                error = error + 1;
                break;
            end
        end
    end
    
    for i = 1:size(b,1)
        for j = 1:size(n_aB,1)
            if(get_distance(b(i,:), rand_points_B(j+1,:)) - get_distance(b(i,:), rand_points_A(j+1,:)) < 0 && n_aB(j) == 0)
                break;
            elseif(get_distance(b(i,:), rand_points_B(j+1,:)) - get_distance(b(i,:), rand_points_A(j+1,:)) > 0 && n_bA(j) == 0 || (j == size(n_aB,1)))
                error = error + 1;
                break;
            end
        end
    end
end
    
function discriminant = create_discriminant(X1, X2, classifiers, n_aB, n_bA)  
    discriminant = zeros(size(X1,1), size(X2,2));  
    for i = 1:size(X1,1)
        for j = 1:size(X2,2)
            for g = 1:size(n_aB)
                G = classifiers(:,:,g+1);
                if(G(i,j) > 0 && n_aB(g) == 0)
                    discriminant(i,j) = 1; % Class B
                    break;
                elseif (G(i, j) < 0 && n_bA(g) == 0)
                    discriminant(i,j) = -1; % Class A
                    break;
                end
            end
        end
    end
end

function dist = get_distance(x1, x2)
    dist = sqrt((x1-x2)*(x1-x2)');
end

function [MED] = get_MED(X, Y, mean1, mean2)
    MED = zeros(size(X,1), size(Y,2));

    for i = 1:size(X, 1)
        for j = 1:size(Y, 2)
            point = [X(i,j) Y(i,j)];

            % if < 0, belongs to class 1; if > 0, belongs to class 2
            MED(i, j) = get_distance(point, mean1) - get_distance(point, mean2);
        end 
    end      
end

function [A, B] = classification_error(a, b, prototypeA, prototypeB)
    MED_a = zeros([size(a,1),3]);
    MED_b = zeros([size(b,1),3]);
    
    for i = 1:size(a,1)
        pointA = a(i,:);
        MED_a(i,:) = [get_distance(pointA, prototypeA) - get_distance(pointA, prototypeB), pointA];
    end
    
    for i = 1:size(b,1)
        pointB = b(i,:);
        MED_b(i,:) = [get_distance(pointB, prototypeB) - get_distance(pointB, prototypeA), pointB];
    end

    % Save a copy of points that were incorrectly classified
    misclassified_indices_A = find(MED_a(:,1) > 0);
    misclassified_indices_B = find(MED_b(:,1) > 0);
    
    A = MED_a(misclassified_indices_A, :);
    B = MED_b(misclassified_indices_B, :);
    
    % Delete the first column that holds difference in distance
    A(:,1) = [];
    B(:,1) = [];
end