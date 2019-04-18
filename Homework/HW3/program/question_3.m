%% Machine Learning Homework 3 Question 3
% Author: Xinrun Zhang
% Time: 02/28/2019 13:42
% =====================================================================

%% initializting
% gradient = A * x[i] + b;
A = [6 -2; -2 6];
b = [-1; -1];

% initial guess x, learning rate alpha and iteration
x = [0; 0];
alpha = 0.1;
itr = 30;
% =====================================================================

%% steepest descent
for i = 1:itr
    x = x - alpha * (A * x + b);
    fprintf('iteration time: %d, x%d = [%f; %f].\n',i,i,x(1),x(2));
end
% =====================================================================

%% plot
[x1, x2] = meshgrid(-1:0.02:1);
z = 3 * (x1.^2) + 3 * (x2.^2) - 2 * x1 * x2 - x1 - x2;
surf(x1, x2, z);
% =====================================================================
