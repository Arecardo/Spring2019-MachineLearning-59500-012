%% Machine Learning Homework 5 part 1
% Author: Xinrun Zhang
% Time: 03/23/2019 17:20
% =====================================================================

%% Initialization
clear ; close all; clc

fprintf('Initializing...\n')
% Initial the data
p = [3 1; 3 2; 4 1; 1 5; 2 4; 2 5];
t = [1; 1; 1; 0; 0; 0];

% Initial the theta vector
theta = [0; 0; 0];

% Initial the learning rate and iteration times
alpha = 1;
iteration = 10;

% Data processing
fprintf('Data processing...\n\n')
p = [ones(6, 1), p(:,1:2)];
% =====================================================================

%% Training the neuron
fprintf('Start training the neuron...\n')
i = 0;
for i = 1:iteration
    theta = trainingNueron( theta, p, t, alpha);
end

fprintf('\nAfter training with alpha = 0.1, ')
fprintf('\nTheta found by training:\n');
fprintf('%.2f\n', theta);
fprintf('\n')
% =====================================================================

%% Prediction
u1 = [1; 1; 4];
u2 = [1; 4; 2];

predict1 = round(logsig( u1'*theta ));
predict2 = round(logsig( u2'*theta ));

fprintf('For U1 = [%d, %d], predict = %d\n',u1(2), u1(3), predict1);
fprintf('For U1 = [%d, %d], predict = %d\n',u2(2), u2(3), predict2);
% =====================================================================

%% Plot
x1 = [3; 3; 4]; y1 = [1; 2; 1];
x2 = [1; 2; 2]; y2 = [5; 4; 5];
x3 = [1; 4];  y3 = [4; 2];
m = 0:0.1:4.5;
n = 1.5*m + 0.25;

figure('Name','Data and Decision boundary','NumberTitle','off');
scatter(x1, y1, 80, 'o', 'r');
hold on;
scatter(x2, y2, 80, 'o', 'b');
hold on;
scatter(x3, y3, 80, 'x', 'g')
hold on;
plot(m, n);
hold off;
legend('1', '0', 'Given patterns', 'Decision boundary');
% =====================================================================
