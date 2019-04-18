%% Machine Learning Homework 6 part 3
% Author: Xinrun Zhang
% Time: 04/02/2019 16:52
% =====================================================================

%% Initializing
clear ; close all; clc

fprintf('Initializing...\n')
% Initial the data
x = [1,1; 1,2; 2,-1; 2,0; -1,2; -2,1; -1,-1; -2,-2];
t = [-1,-1; -1,-1; -1,1; -1,1; 1,-1; 1,-1; 1,1; 1,1];
[m, n] = size(x);

% Initial the theta with bias
theta = [1,1; 1,0; 0,1];

%Initial the learning rate
alpha = 0.02;

% Initial the iteration and 
% initial errors. Set errors to 1 to start the iteration
itr = 10000;
e = zeros(m,2);


% Data processing
X = [ones(m,1), x(:,1:2)];
% =====================================================================

%% LMS algorithm
fprintf('Starting the algorithm...\n');
for iteration = 1:itr
    for i = 1:m
        v = X(i,:) * theta;
        e(i, :) = t(i,:) - v;
        theta = theta + 2 * alpha * X(i,:)' * e(i,:);
    end
end

p = zeros(8,2);
for i = 1:m
    p(i,:) = X(i,:) * theta;
    if p(i,1) >= 0
        p(i,1) = 1;
    else
        p(i,1) = -1;
    end
    
    if p(i,2) >= 0 
        p(i,2) = 1;
    else
        p(i,2) = -1;
    end
end
fprintf('\nAfter training with alpha = 0.2, ')
fprintf('\nTheta found by training after %d iterations, which means calculated %d * %d times:\n', itr, itr, m);
fprintf('%.2f %.2f\n', theta');
fprintf('\n')

fprintf('The 2-norm of theta is %.2f\n', norm(theta))
fprintf('\nThe predicted output is:\n')
fprintf('%.2f %.2f\n', p')

% =====================================================================

%% plot the data
a = -2.4:0.1:2.4;
figure('Name','Data and Decision boundary','NumberTitle','off');
scatter(x(1:2,1), x(1:2,2), 80, 'o', 'r');
hold on;
scatter(x(3:4,1), x(3:4,2), 80, 'x', 'b');
hold on;
scatter(x(5:6,1), x(5:6,2), 80, '+', 'k');
hold on;
scatter(x(7:8,1), x(7:8,2), 80, '*', 'm');
hold on;
plot(a, (-0.57/0.02)*a + 1, '-');
hold on;
plot(a, (0.2/0.65)*a + 0.2/0.65, '-');
xlim([-2.4, 2.4]);
ylim([-2.4, 2.4]);
legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Decision boundary 1', 'Decision boundary 2');
% =====================================================================
