%% Machine Learning Homework 6 part 1 and 2
% Author: Xinrun Zhang
% Time: 04/02/2019 11:04
% =====================================================================

%% Initializing
clear ; close all; clc

fprintf('Initializing...\n')
% Initial the data
x1 = [1; 1];
x2 = [1; -1];
t1 = 1;
t2 = -1;
p = 0.5;

% Calculate c, h and R, where
% c = E[t.^2]
% h = E[t .* x]
% R = E[x * x']
c = p * t1^2 + p * t2^2;
h = p * (t1 * x1) + p * (t2 * x2);
R = p * (x1 * x1') + p * (x2 * x2');


% Calculate the optimal theta
theta_opt = (R^-1)* h;

% Calculate the Hessian
H = 2*R;
eig_value = eig(H);
% =====================================================================

%% LMS algorithm
% Initial the x and t
x = [1, 1; 1, -1];
t = [1; -1];
m = 2;

% Initial the theta and the learning rate
theta = [0; 0];
alpha = 0.2;

% Initial the iteration and the error
itr = 0;
e = [0;0];


% LMS algorithm
fprintf('Starting the algorithm...\n');
while ( 1 )
    for i = 1:m
        v = x(i,:) * theta;
        e(i) = t(i) - v;
        theta = theta + 2 * alpha * e(i) * x(i,:)';
    end
    itr = itr + 1;
    if ~(any(any(e)))
        break;
    end
end

fprintf('\nAfter training with alpha = 0.2, ')
fprintf('\nTheta found by training after %d iterations:\n', itr);
fprintf('%.2f\n', theta);
fprintf('\n') 
% =====================================================================

%% plot the data
a = 0.8:0.1:1.2;
figure('Name','Data and Decision boundary','NumberTitle','off');
scatter(x1(1), x1(2), 80, 'o', 'r');
hold on;
scatter(x2(1), x2(2), 80, 'x', 'b');
hold on;
plot(a, 0*a, '-');
ylim([-1.3, 1.3]);
legend('x1', 'x2','Decision boundary');
% =====================================================================
