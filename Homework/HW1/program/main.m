%% Machine Learning Homework 1
% Author: Xinrun Zhang
% Time: 02/09/2019 18:45
% =====================================================================

%% Initialization
clear ; close all; clc

% Import the data;
fprintf('Initializing...\n')
fprintf('Reading the data...\n');
A = xlsread('AutoData_HW1.xlsx');

% Extract the column 4 as input x and column 6 as input y;
x = A(:,4);
y = A(:,6);

% Plot the data;
fprintf('Visualizing the data...\n\n')
figure('Name','Raw Data','NumberTitle','off');
plot(x,y,'x','MarkerSize',8);
ylabel('MPG');
xlabel('Weight');

% Initialize the theta vector;
theta = zeros(2, 1);

% Initialize the gradient descent parameters
iteration = 2000;
alpha = 0.01;
% =====================================================================

%% Data processing
fprintf('Data processing...\n')
% choose normalizing the data or not;
choice = input('Do you want to normalize the data? 1.y / 0.n\n');
switch choice
    case 1
        x_nor = normalizing(x); % Call the normalizing function;
        X = [ones(50, 1), x_nor(:,1)]; % Add a column of ones to xnor;
    case 0
        X = [ones(50, 1), x(:,1)]; % Add a column of ones to x;
end
% =====================================================================

%% Univariable linear regression
J = computeCost(X, y, theta); %compute the cost 
fprintf('\nWith theta = [0 ; 0]\nCost computed = %f\n', J);

% running gradient descent
theta = gradientDescent(X, y, theta, iteration, alpha);
% print the output, including new theta and J;
fprintf('\nTheta found by gradient descent:\n');
fprintf('%f\n', theta);
J = computeCost(X, y, theta);
fprintf('\nWith theta = [%f ; %f]\nCost computed = %f\n', theta(1),theta(2), J);

figure('Name','Univariable Linear Regression','NumberTitle','off');
plot(x_nor,y,'x','MarkerSize',8);
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-') % plot the hypothesis
legend('Training data', 'Linear regression');

x_predict = theta(1) + theta(2)*((3500 - mean(x))/(max(x) - min(x)));
fprintf('\nWith x = 3100, the predict y = %f\n',x_predict);
fprintf('Now, the program is paused.\n');
fprintf('If you want to go to polynomial regression,\n')
fprintf('please press enter.\n')
pause;
% =====================================================================

%% Polynomial Regression with feature scaling
% initialization
x_pyn = [x, x.^2, x.^3]; % raw data matrix;
theta_pyn = zeros(4, 1);
iteration_pyn = 2000;
alpha_pyn = 0.01;
x_predict = 3100;

% Feature scaling;
[X_pyn, x_predict]= featureScaling(x_pyn, x_predict);
X_pyn = [ones(50, 1), X_pyn(:,1), X_pyn(:,2), X_pyn(:,3)];

% Gradient descent;
theta_pyn = gradientDescentPyn(X_pyn, y, theta_pyn, iteration_pyn, alpha_pyn);

figure('Name','Polynomial Regression','NumberTitle','off');
plot(X_pyn(:,2),y,'x','MarkerSize',8);
hold on;
a = -1.8:0.01:2;
plot(a, theta_pyn(1)+theta_pyn(2)*a+theta_pyn(3)*(a.^2)+theta_pyn(4)*(a.^3), '-');

% print the result theta;
fprintf('\nTheta found by gradient descent:\n');
fprintf('%f\n', theta_pyn);
J = computeCost(X_pyn, y, theta_pyn);
fprintf('\nWith theta = [%f ; %f; %f; %f]\nCost computed = %f\n', theta_pyn(1),theta_pyn(2),theta_pyn(3),theta_pyn(4), J);

% compute the x_predict and print the result;
x_predict = theta_pyn(1) + theta_pyn(2)*(x_predict) + theta_pyn(3)*(x_predict.^2) + theta_pyn(4)*(x_predict.^3);
fprintf('\nWith x = 3100, the predict y = %f\n',x_predict');

% =====================================================================