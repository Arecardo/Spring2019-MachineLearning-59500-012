%% Machine Learning Homework 4 part 2
% Author: Xinrun Zhang
% Time: 03/19/2019 17:28
% =====================================================================
%% Initialization
clear ; close all; clc

% Import the data;
fprintf('Initializing...\n')
fprintf('Reading the data...\n');
data = load('ex2data2.txt');

x = data(:, [1, 2]); 
y = data(:, 3);  % y has values of 0 and 1.

% Initialize the theta vector;
theta = zeros(28, 1);

% Initialize the gradient descent parameters
iteration = 500;
alpha = 0.1;
lambda = 0.5;
% =====================================================================

%% Plot the original data;
data_sorted = sortrows(data,3);
X_0 = data_sorted(1:60, [1, 2]);
X_1 = data_sorted(61:118, [1, 2]);

fprintf('Visualizing the original data...\n\n');
figure('Name','Original Data','NumberTitle','off');
scatter(X_0(:, 1), X_0(:, 2), 'o', 'r');
hold on;
scatter(X_1(:, 1), X_1(:, 2), 'x', 'b');
hold off;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('Not admitted', 'admitted');
% =====================================================================
%% Data processing
fprintf('Data processing...\n\n')
% rebuild the x
% 1 1
% 2 11 2
% 3 21 12 3
% 4 32 23 4
% 5 43 42 24 34 5
% 6 54 53 52 25 35 45 6
x_2 = [x(:,1).^2, x(:,1).*x(:,2), x(:,2).^2];
x_3 = [x(:,1).^3, (x(:,1).^2).*x(:,2), x(:,1).*(x(:,2).^2), x(:,2).^3 ];
x_4 = [x(:,1).^4, (x(:,1).^3).*(x(:,2).^2), (x(:,1).^2).*(x(:,2).^3), x(:,2).^4];
x_5 = [x(:,1).^5, (x(:,1).^4).*(x(:,2).^3), (x(:,1).^4).*(x(:,2).^2), (x(:,1).^2).*(x(:,2).^4), (x(:,1).^3).*(x(:,2).^4), x(:,2).^5];
x_6 = [x(:,1).^6, (x(:,1).^5).*(x(:,2).^4), (x(:,1).^5).*(x(:,2).^3), (x(:,1).^5).*(x(:,2).^2), (x(:,1).^2).*(x(:,2).^5), (x(:,1).^3).*(x(:,2).^5), (x(:,1).^4).*(x(:,2).^5), x(:,2).^6];
x = [x(:, 1:2), x_2(:, 1:3), x_3(:, 1:4), x_4(:, 1:4), x_5(:, 1:6), x_6(:, 1:8)];


% normalization
%x = normalization(x);

% Add a column of ones to x;
X = [ones(118, 1), x(:,1:27)];
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Logistic regression
fprintf('Starting logistic regression...');
J = computeCost_part2(X, y, theta, lambda); %compute the cost 
fprintf('\nWith theta = zeros(28, 1)\nCost computed = %f\n', J);

% Running gradient descent
theta = gradientDescent_part2(X, y, theta, iteration, alpha, lambda);

% Print the output, including new theta and J;
fprintf('\nAfter 500 iterations with alpha = 0.1 and lambda = %f, ', lambda);
fprintf('\nTheta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X, y, theta);
fprintf('\nWith this theta, \nCost computed = %f\n',J);
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Plot the decision boundary
% define two arrays
u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);

% g_2 = [u.^2, u.*v, v.^2];
% g_3 = [u.^3, (u.^2).*v, u.*(v.^2), v.^3 ];
% g_4 = [u.^4, (u.^3).*(v.^2), (u.^2).*(v.^3), v.^4];
% g_5 = [u.^5, (u.^4).*(v.^3), (u.^4).*(v.^2), (u.^2).*(v.^4), (u.^3).*(v.^4), v.^5];
% g_6 = [u.^6, (u.^5).*(v.^4), (u.^5).*(v.^3), (u.^5).*(v.^2), (u.^2).*(v.^5), (u.^3).*(v.^5), (u.^4).*(v.^5), v.^6];
% g = [u, v, g_2(:, 1:3), g_3(:, 1:4), g_4(:, 1:4), g_5(:, 1:6), g_6(:, 1:8)];

z = zeros(length(u), length(v));
% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = mapFeature(u(i), v(j))*theta;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% need to specify the range [0, 0]
figure('Name','data with Decision boundary','NumberTitle','off');
scatter(X_0(:, 1), X_0(:, 2), 'o', 'r');
hold on;
scatter(X_1(:, 1), X_1(:, 2), 'x', 'b');
hold on;
contour(u, v, z, [0, 0])
hold off;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('Not admitted', 'admitted', 'decision boundary');
% =====================================================================

%% compute the accuracy
predict = round(logsig(X * theta));

accuracy = mean( double(predict == y) * 100);

fprintf('The accuracy is %f\n', accuracy);
fprintf('---------------------------------------------------------\n');
% =====================================================================
