%% Machine Learning Homework 3 Question 6
% Author: Xinrun Zhang
% Time: 02/28/2019 14:03
% =====================================================================

%% initializting
% gradient = 2 * A * x
% Hessian = 2 * A
A = [7/2 -3; -3 1];

H = 2 * A;
H_inv = (H)^-1;

% initial guess x
x = [1; 1];
% =====================================================================

%% Newton's Method
x = x - H_inv * (2 * A * x);
fprintf('x found by Newtons Method is: [%f; %f]\n',x(1), x(2));
% =====================================================================

%% plot
[x1, x2] = meshgrid(-1:0.03:1);
z = (7/2) * (x1.^2) - (x2.^2) - 6 * x1 * x2;
surf(x1, x2, z);
% =====================================================================
