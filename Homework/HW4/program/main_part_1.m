%% Machine Learning Homework 4 part 1
% Author: Xinrun Zhang
% Time: 03/15/2019 14:39
% =====================================================================

%% Initialization
clear ; close all; clc

% Import the data;
fprintf('Initializing...\n')
fprintf('Reading the data...\n');
data = load('ex2data1.txt');

X = data(:, [1, 2]); 
y = data(:, 3);  % y has values of 0 and 1.

% Initialize the theta vector;
theta = zeros(3, 1);

% Initialize the gradient descent parameters
iteration = 400;
alpha = 0.1;
% =====================================================================

%% Plot the original data;
data_sorted = sortrows(data,3);
X_0 = data_sorted(1:40, [1, 2]);
X_1 = data_sorted(41:100, [1, 2]);

fprintf('Visualizing the original data...\n\n');
figure('Name','Original Data','NumberTitle','off');
scatter(X_0(:, 1), X_0(:, 2), 'o', 'r');
hold on;
scatter(X_1(:, 1), X_1(:, 2), 'x', 'b');
hold off;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Not admitted', 'admitted');
% =====================================================================

%% Data processing
fprintf('Data processing...\n\n')
% Add a column of ones to x;
x = [ones(100, 1), X(:,1:2)];
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Logistic regression version 1
fprintf('Starting logistic regression version 1...');
J = computeCost(x, y, theta); %compute the cost 
fprintf('\nWith theta = [0; 0; 0]\nCost computed = %f\n', J);

% Running gradient descent
theta = gradientDescent(x, y, theta, iteration, alpha);
% Print the output, including new theta and J;
fprintf('\nAfter 400 iterations with alpha = 0.1, ')
fprintf('\nTheta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(x, y, theta);
fprintf('\nWith theta = [%f ; %f; %f]\nCost computed = %f\n', theta(1),theta(2),theta(3), J);
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Analysis
% I can't get proper desicion boundary from gradient descent algorithm
% which I wrote in the function gradientDescent.m.
% Therefore, I searched online and found another way to get the optimal
% The computeCost_new.m is created to generate gradient.
% =====================================================================

%% Logistic regression version 2
fprintf('Starting logistic regression version 2...');
theta_new = zeros(3, 1);
[~, grad] = computeCost_new(theta_new, x, y);

% run the function optimization algorithm
options = optimset('GradObj', 'on', 'MaxIter', 400);
theta_new= fminunc(@(t)computeCost_new(t, x, y), theta_new, options);

fprintf('\nTheta found by optimization algorithm:\n');
fprintf('%f\n', theta_new);

J = computeCost(x, y, theta_new);
fprintf('\nWith theta_v2 = [%f ; %f; %f]\nCost computed = %f\n', theta_new(1),theta_new(2),theta_new(3), J);
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Plot the decision boundary
a = 30:0.1:100;
db_1 = (-1./theta(3)).*(theta(2).*a + theta(1));
db_2 = (-1./theta_new(3)).*(theta_new(2).*a + theta_new(1));
figure('Name','Original Data','NumberTitle','off');
scatter(X_0(:, 1), X_0(:, 2), 'o', 'r');
hold on;
scatter(X_1(:, 1), X_1(:, 2), 'x', 'b');
hold on;
plot(a, db_1);
hold on;
plot(a,db_2);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Not admitted', 'admitted', 'Decision Boundary 1', 'Decision Boundary 2');
% =====================================================================

%% Compute the accuracy
predict_v1 = round(logsig(x * theta));
predict_v2 = round(logsig(x * theta_new));

accuracy_v1 = mean( double(predict_v1 == y) * 100);
accuracy_v2 = mean( double(predict_v2 == y) * 100);

fprintf('For version 1, the accuracy is %f\n', accuracy_v1);
fprintf('For version 2, the accuracy is %f\n', accuracy_v2);
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Compute the probability
prob = logsig(theta_new(1) + theta_new(2)*45 + theta_new(3)*85);
fprintf('The probability of this student getting admitted is %f\n', prob);
% =====================================================================
