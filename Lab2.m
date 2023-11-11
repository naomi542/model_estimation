clc;
close all;
clear all; 
%% Load datasets
% There's probs a better way to do this looll

% Part 1. 1D Case
classes1 = load('data/lab2_1.mat');
A1 = classes1.a';   % I transposed so its a col vector
B1 = classes1.b';   % trasnposed to col vector

% Part 2. 2D Case
% A2, B2, C2 refer to training
% A2_test, B2_test, C2_test refer to testing
classes2 = load('data/lab2_2.mat');
A2 = classes2.al;
B2 = classes2.bl;
C2 = classes2.cl;

% Part 3. Sequential Classifier
classes3 = load('data/lab2_3.mat');
A3 = classes3.a;
B3 = classes3.b;

%% 1. Model Estimation 1D Case

% Gaussian Parametric Estimation
[mu_A1, sigma_A1] = gaussian_param_est(A1);
[mu_B1, sigma_B1] = gaussian_param_est(B1);

x_A = 0:0.05:max(A1)+1;
x_B = 0:0.05:max(B1)+1;

true_A = normpdf(x_A,5,1);
true_B = exppdf(x_B,1);

% Plot True vs Estimated Gaussian of A
figure('Name', 'Gaussian Parametric Estimation of A');
set(gcf,'color','w');
hold on
plot(x_A, true_A, 'DisplayName', 'True Distribution');
plot(x_A, normpdf(x_A, mu_A1, sigma_A1), 'DisplayName', 'Estimated Distribution');
scatter(A1, zeros(size(A1)), 'DisplayName', 'Class A Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

% Plot True vs Estimated Gaussian of B
figure('Name', 'Gaussian Parametric Estimation of B');
set(gcf,'color','w');
hold on
plot(x_B, true_B, 'DisplayName', 'True Distribution');
plot(x_B, normpdf(x_B, mu_B1, sigma_B1), 'DisplayName', 'Estimated Distribution');
scatter(B1, zeros(size(B1)), 'DisplayName', 'Class B Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

%%
% Exponential Parametric Estimation
lambdaA1 = exponential_1d(A1);
lambdaB1 = exponential_1d(B1);

% Plot True vs Estimated Exponential of A
figure('Name', 'Exponential Parametric Estimation of A');
set(gcf,'color','w');
hold on
plot(x_A, true_A, 'DisplayName', 'True Distribution');
plot(x_A, exppdf(x_A, 1/lambdaA1), 'DisplayName', 'Estimated Distribution');
scatter(A1, zeros(size(A1)), 'DisplayName', 'Class A Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

% Plot True vs Estimated Exponential of B
figure('Name', 'Gaussian Parametric Estimation');
set(gcf,'color','w');
hold on
plot(x_B, true_B, 'DisplayName', 'True Distribution');
plot(x_B, exppdf(x_B, 1/lambdaB1), 'DisplayName', 'Estimated Distribution');
scatter(B1, zeros(size(B1)), 'DisplayName', 'Class B Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off
%%
% Uniform Parametric Estimation
[minA1, maxA1] = uniform_1d(A1);
[minB1, maxB1] = uniform_1d(B1);

% Plot True vs Estimated Uniform of A
figure('Name', 'Uniform Parametric Estimation of A');
set(gcf,'color','w');
hold on
plot(x_A, true_A, 'DisplayName', 'True Distribution');
plot(x_A, unifpdf(x_A, minA1, maxA1), 'DisplayName', 'Estimated Distribution');
scatter(A1, zeros(size(A1)), 'DisplayName', 'Class A Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

% Plot True vs Uniform Exponential of B
figure('Name', 'Uniform Parametric Estimatio n of B');
set(gcf,'color','w');
hold on
plot(x_B, true_B, 'DisplayName', 'True Distribution');
plot(x_B, unifpdf(x_B, minB1, maxB1), 'DisplayName', 'Estimated Distribution');
scatter(B1, zeros(size(B1)), 'DisplayName', 'Class B Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off
%%
% Parzen 1D
pHat1_A1 = parzen1D(x_A, A1, 0.1);
pHat4_A1 = parzen1D(x_A, A1, 0.4);
pHat1_B1 = parzen1D(x_B, B1, 0.1);
pHat4_B1 = parzen1D(x_B, B1, 0.4);

% Plot True vs Parzen of A; std = 0.1 
figure('Name', 'Parzen Estimation of A - std = 0.1');
set(gcf,'color','w');
hold on
plot(x_A, true_A, 'DisplayName', 'True');
plot(x_A, pHat1_A1, 'DisplayName', 'Estimated');
scatter(A1, zeros(size(A1)), 'DisplayName', 'Class A');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

% Plot True vs Parzen of A; std = 0.4
figure('Name', 'Parzen Estimation of A - std = 0.4');
set(gcf,'color','w');
hold on
plot(x_A, true_A, 'DisplayName', 'True Distribution');
plot(x_A, pHat4_A1, 'DisplayName', 'Estimated Distribution');
scatter(A1, zeros(size(A1)), 'DisplayName', 'Class A Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

% Plot True vs Estimated Exponential of B
figure('Name', 'Parzen Estimation of B - std = 0.1');
set(gcf,'color','w');
hold on
plot(x_B, true_B, 'DisplayName', 'True Distribution');
plot(x_B, pHat1_B1, 'DisplayName', 'Estimated Distribution');
scatter(B1, zeros(size(B1)), 'DisplayName', 'Class B Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

% Plot True vs Estimated Exponential of B
figure('Name', 'Parzen Estimation of B - std = 0.4');
set(gcf,'color','w');
hold on
plot(x_B, true_B, 'DisplayName', 'True Distribution');
plot(x_B, pHat4_B1, 'DisplayName', 'Estimated Distribution');
scatter(B1, zeros(size(B1)), 'DisplayName', 'Class B Samples');
legend
xlabel("Feature 1 (x)")
ylabel("PDF p(x)")
hold off

%% 2. Model Estimation 2D Case

% Create grid for 2D ML Classifier
increment = 1;
ml_x = min([A2(:,1);B2(:,1);C2(:,1)]):increment:max([A2(:,1);B2(:,1);C2(:,1)]);
ml_y = min([A2(:,2);B2(:,2);C2(:,2)]):increment:max([A2(:,2);B2(:,2);C2(:,2)]);
[ML_X, ML_Y] = meshgrid(ml_x,ml_y);

% Gaussian Parametric Estimation
[mu_A2, sigma_A2] = gaussian_param_est(A2);
[mu_B2, sigma_B2] = gaussian_param_est(B2);
[mu_C2, sigma_C2] = gaussian_param_est(C2);

ML_AB = get_ML(ML_X, ML_Y, mu_A2, mu_B2, sigma_A2, sigma_B2);
ML_BC = get_ML(ML_X, ML_Y, mu_B2, mu_C2, sigma_B2, sigma_C2);
ML_AC = get_ML(ML_X, ML_Y, mu_A2, mu_C2, sigma_A2, sigma_C2);

ML_ABC = classify_points(ML_X, ML_Y, ML_AB, ML_BC, ML_AC);

% Plot samples, std contours, MED decision boundaries
figure('Name','ML Classifier A2, B2, C2');
set(gcf,'color','w');
hold on

contour(ML_X, ML_Y, ML_ABC, [-1 0 1], 'k', 'LineWidth', 2, 'DisplayName', 'ML Decision Boundary');
scatter(A2(:,1), A2(:,2), 'filled', 'DisplayName', 'Class A','MarkerFaceColor', [171/255 111/255 169/255]);     
scatter(B2(:,1), B2(:,2), 'filled', 'DisplayName', 'Class B','MarkerFaceColor', [29/255 146/255 64/255]);    
scatter(C2(:,1), C2(:,2), 'filled', 'DisplayName', 'Class C','MarkerFaceColor', [0 76/255 153/255]);
legend
xlabel("Feature 1")
ylabel("Feature 2")
hold off
%%
% Parzen Non-Parametric Estimation
% Create Gaussian window for parzen function
h = 400;
step = 1;
[X2, Y2] = meshgrid(1:step:h);
X = [X2(:) Y2(:)];
mu = [h/2 h/2];
cov_2D = [h 0;0 h];

gaussian_win = mvnpdf(X, mu, cov_2D);
gaussian_win = reshape(gaussian_win, length(Y2), length(X2));

% Create resolution and and parzen PDFs
minx = ml_x(1) - 10;
miny = ml_y(1) - 10;
maxx = ml_x(length(ml_x)) + 10;
maxy = ml_y(length(ml_y)) + 10;
resolution = [step minx miny maxx maxy];

[phat_A2, x_A2, y_A2] = parzen(A2, resolution, gaussian_win);
[phat_B2, x_B2, y_B2] = parzen(B2, resolution, gaussian_win);
[phat_C2, x_C2, y_C2] = parzen(C2, resolution, gaussian_win);

classify_grid= classify_ML(phat_A2, phat_B2, phat_C2);
[scaled_x, scaled_y] = meshgrid(x_A2, y_A2);

figure();
set(gcf,'color','w');
hold on;
contour(scaled_x, scaled_y, classify_grid, 'k', 'LineWidth', 2, 'DisplayName', 'ML Decision Boundary - Parzen');
scatter(A2(:,1), A2(:,2), 'filled', 'DisplayName', 'Class A2','MarkerFaceColor', [171/255 111/255 169/255]);     
scatter(B2(:,1), B2(:,2), 'filled', 'DisplayName', 'Class B2','MarkerFaceColor', [29/255 146/255 64/255]);    
scatter(C2(:,1), C2(:,2), 'filled', 'DisplayName', 'Class C2','MarkerFaceColor', [0 76/255 153/255]);
legend
xlabel("Feature 1")
ylabel("Feature 2")
hold off;

%% 4. Sequential
x1 = min([A3(:,1);B3(:,1)]):1:max([A3(:,1);B3(:,1)]);
x2 = min([A3(:,2);B3(:,2)]):1:max([B3(:,2);B3(:,2)]);
[X1, X2] = meshgrid(x1,x2);

% Unlimited number of classifiers
sequential_discriminant_1 = sequential_discriminant(A3, B3);

figure('Name','Sequential Classifier #1');
set(gcf,'color','w');
hold on;
scatter(A3(:,1), A3(:,2), 'filled', 'DisplayName', 'Class A','MarkerFaceColor', [171/255 111/255 169/255]);     
scatter(B3(:,1), B3(:,2), 'filled', 'DisplayName', 'Class B','MarkerFaceColor', [29/255 146/255 64/255]);    
contour(X1, X2, sequential_discriminant_1, 'k', 'LineWidth', 2, 'DisplayName', 'Sequential Discriminant');
legend
xlabel("Feature 1")
ylabel("Feature 2")
hold off;

% Unlimited number of classifiers
sequential_discriminant_2 = sequential_discriminant(A3, B3);

figure('Name','Sequential Classifier #2');
set(gcf,'color','w');
hold on;
scatter(A3(:,1), A3(:,2), 'filled', 'DisplayName', 'Class A','MarkerFaceColor', [171/255 111/255 169/255]);     
scatter(B3(:,1), B3(:,2), 'filled', 'DisplayName', 'Class B','MarkerFaceColor', [29/255 146/255 64/255]);    
contour(X1, X2, sequential_discriminant_2, 'k', 'LineWidth', 2, 'DisplayName', 'Sequential Discriminant');
legend
xlabel("Feature 1")
ylabel("Feature 2")
hold off;

% Unlimited number of classifiers
sequential_discriminant_3 = sequential_discriminant(A3, B3);

figure('Name','Sequential Classifier #3');
set(gcf,'color','w');
hold on;
scatter(A3(:,1), A3(:,2), 'filled', 'DisplayName', 'Class A','MarkerFaceColor', [171/255 111/255 169/255]);     
scatter(B3(:,1), B3(:,2), 'filled', 'DisplayName', 'Class B','MarkerFaceColor', [29/255 146/255 64/255]);    
contour(X1, X2, sequential_discriminant_3, 'k', 'LineWidth', 2, 'DisplayName', 'Sequential Discriminant');
legend
xlabel("Feature 1")
ylabel("Feature 2")
hold off;
%%
error = zeros(5, 20);
avg_errors = zeros(5,1);
min_errors = zeros(5,1);
max_errors = zeros(5,1);
std_errors = zeros(5,1);

for i = 1:5
    for j = 1:20
         error(i, j) = sequential_discriminant(A3, B3, i, true)/400;     
    end
    
    avg_errors(i) = mean(error(i,:));
    min_errors(i) = min(error(i,:));
    max_errors(i) = max(error(i,:));
    std_errors(i) = std(error(i,:));
end

x_axis = linspace(1,5,5);
figure();
set(gcf,'color','w');
hold on;
line(x_axis, avg_errors, 'DisplayName', 'Average Error', 'Color', 'k');
line(x_axis, min_errors, 'DisplayName', 'Minimum Error', 'Color', 'b');
line(x_axis, max_errors, 'DisplayName', 'Maximum Error', 'Color', 'r');
line(x_axis, std_errors, 'DisplayName', 'Standard Devation of Error', 'Color', 'g');
legend
xlabel("Number of Classifiers")
ylabel("Error Rate")
xticks([1,2,3,4,5])
hold off;
