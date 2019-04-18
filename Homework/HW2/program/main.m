%% Machine Learning Homework 2
% Author: Xinrun Zhang
% Time: 02/20/2019 14?39
% =====================================================================

%% Initialization
clear ; close all; clc

% Import the data;
fprintf('Initializing...\n')
fprintf('Reading the data...\n');
A = xlsread('AutoData_HW2.xlsx');

% Extract the data, generate training data and validation data;
x = A(1:160,2:5);
y = A(1:160,1);
x_val = A(161:200,2:5);
y_val = A(161:200,1);

% Plot the data;
fprintf('Visualizing the data...\n\n')
figure('Name','Raw Data','NumberTitle','off');
plot(x(:,4),y,'x','MarkerSize',8);
ylabel('MPG');
xlabel('Weights');

% Initialize the theta vector;
theta = zeros(5, 1);

% Initialize the gradient descent parameters
iteration = 1000;
alpha = 0.01;
% =====================================================================

%% Data processing
fprintf('Data processing...\n')

% Call the normalizing function;
x_nor = normalizing(x);
x_val_nor = normalizing(x_val);

% Add a column of ones to xnor;
X = [ones(160, 1), x_nor(:,1:4)];
X_val = [ones(40, 1), x_val_nor(:,1:4)];

% =====================================================================

%% Multivariate linear regression
J = computeCost(X, y, theta); %compute the cost 
fprintf('\nWith theta = [0 ; 0]\nCost computed = %f\n', J);

% Running gradient descent
theta = gradientDescent(X, y, theta, iteration, alpha);
% Print the output, including new theta and J;
fprintf('\nTheta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X, y, theta);
fprintf('\nWith theta = [%f ; %f; %f; %f; %f]\nCost computed = %f\n', theta(1),theta(2),theta(3),theta(4),theta(5), J);
% =====================================================================

%% Validation
% Generate predict data
y_prd = X_val*theta;

% Calculate the mean squared error err_cv
err_cv = immse(y_prd,y_val);
fprintf('The mean suqare error err_cv is: %f\n',err_cv);

% Plot of predicted MPG against observed MPG for the validation data
figure('Name','Predicted MPG vs Observed MPG','NumberTitle','off');
plot(x_val(:,4),y_prd,'x','MarkerSize',8);
ylabel('MPG');
xlabel('Weights');
hold on;
plot(x_val(:,4),y_val,'o','MarkerSize',8);
% =====================================================================

%% Normal equation
% Data_processing
X_ne = [ones(160,1), x(:,1:4)];
X_ne_val = [ones(40,1), x_val(:,1:4)];
% Normal equation
theta_ne = pinv(X_ne)* y;
%theta_ne = (X'*X)^(-1)* X' * y;
y_prd_ne = X_ne_val*theta_ne;

% Print the output, including new theta and J;
fprintf('\nTheta found by normal equation:\n');
fprintf('%f\n', theta_ne);

J = computeCost(X_ne, y, theta_ne);
fprintf('\nWith theta = [%f ; %f; %f; %f; %f]\nCost computed = %f\n', theta_ne(1),theta_ne(2),theta_ne(3),theta_ne(4),theta_ne(5), J);

% Plot of predicted MPG against observed MPG for the validation data
figure('Name','Predicted MPG vs Observed MPG 2','NumberTitle','off');
plot(x_val(:,4),y_prd_ne,'x','MarkerSize',8);
ylabel('MPG');
xlabel('Weights');
hold on;
plot(x_val(:,4),y_val,'o','MarkerSize',8);
% =====================================================================

%% Optional exercise
% Extract the data, generate training data and validation data;
x_oe = A(1:160,2:6);
y_oe = A(1:160,1);
x_oe_val = A(161:200,2:6);
y_oe_val = A(161:200,1);

% Initialize the theta vector;
theta_oe = zeros(6, 1);

% Initialize the gradient descent parameters
iteration = 1000;
alpha = 0.01;
% =====================================================================

% Data processing
x_oe_nor = normalizing(x_oe);
X_oe = [ones(160, 1), x_oe_nor(:,1:5)];

x_oe_val_nor = normalizing(x_oe_val);
X_oe_val = [ones(40, 1), x_oe_val_nor(:,1:5)];

% Gradient Descent
theta_oe = gradientDescent(X_oe, y, theta_oe, iteration, alpha);

% Print the output, including new theta and J;
fprintf('\nOptional exercise Theta found by gradient descent:\n');
fprintf('%f\n', theta_oe);

J = computeCost(X_oe, y_oe, theta_oe);
fprintf('\nWith theta = [%f ; %f; %f; %f; %f; %f]\nCost computed = %f\n', theta_oe(1),theta_oe(2),theta_oe(3),theta_oe(4),theta_oe(5), theta_oe(6), J);
% =====================================================================
% Validation
% Generate predict data
y_prd = X_oe_val*theta_oe;

% Plot of predicted MPG against observed MPG for the validation data
figure('Name','Predicted MPG vs Observed MPG','NumberTitle','off');
plot(x_oe_val(:,4),y_prd,'x','MarkerSize',8);
ylabel('MPG');
xlabel('Weights');
hold on;
plot(x_oe_val(:,4),y_val,'o','MarkerSize',8);
% =====================================================================
