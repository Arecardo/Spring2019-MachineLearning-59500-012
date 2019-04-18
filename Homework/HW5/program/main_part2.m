%% Machine Learning Homework 5 part 2
% Author: Xinrun Zhang
% Time: 03/23/2019 21:07
% =====================================================================

%% Initialization
clear ; close all; clc

fprintf('Initializing...\n')
% Initial the data
data = importdata('halfmoon.mat'); % don't use load function
x = data(:,[1, 2]);
y = data(:, 3);
data_val = importdata('halfmoonTest.mat');
x_val = data_val(:,[1, 2]);
y_val = data_val(:, 3);

% Initial the theta vector
theta = [1; 1; 1];

% Initial the learning rate and iteration times
alpha = 1;
iteration = 10;

% Data processing
fprintf('Data processing...\n')
X = [ones(2000, 1), x(:,1:2)];
X_val = [ones(240, 1), x_val(:, 1:2)];
% =====================================================================

%% Training the neuron
fprintf('Start training the neuron...\n\n')
i = 0;
for i = 1:iteration
    theta = trainingNueron( theta, X, y, alpha);
end

fprintf('After training with alpha = 0.1,\n')
fprintf('Theta found by training:\n');
fprintf('%.2f\n', theta);
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Plot the original data
fprintf('Plotting the data...\n')
x_0 = x(1:1000, [1, 2]);
x_1 = x(1001:2000, [1, 2]);
m = -16:0.1:26;
n = 0.0323*m - 0.3322;

figure('Name','Original Data','NumberTitle','off');
scatter(x_0(:, 1), x_0(:, 2), 'o');
hold on;
scatter(x_1(:, 1), x_1(:, 2), 'x');
hold on;
plot(m, n);
legend('0', '1', 'Decision boundary');
fprintf('---------------------------------------------------------\n');
% =====================================================================

%% Validation
predict = round(logsig(X_val*theta));
accuracy = mean( double(predict == y_val) * 100);
fprintf('The accuracy is %.2f\n', accuracy);
fprintf('---------------------------------------------------------\n');
% =====================================================================
