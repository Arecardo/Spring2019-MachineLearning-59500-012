%% Machine Learning Homework 6 part 4
% Author: Xinrun Zhang
% Time: 04/03/2019 14:43
% =====================================================================

%% Initializing
clear ; close all; clc

fprintf('Initializing...\n');
% Import the data
data = importdata('hm.mat');
x = data(:,1:2);
t = data(:,3);
[m,~] = size(x);

% Initial the theta with bias
% theta = ones(3,1);
theta = [-1; -1; -1]; 
fprintf('The initial theta is :\n');
fprintf('%.4f\n', theta);
% If set the theta as all -1, the decision boundary will
% converage from the opposite side

% Initial the learning rate, iteration and error
alpha = 0.0003;
fprintf('The initial learning rate is: alpha = %.4f\n',alpha);
itr = 40;
e = zeros(m,1);

% Data processing
X = [ones(m,1) x(:,1:2)];
t(t==0) = -1; % replace the 0 with -1
% =====================================================================

%% LMS algorithm
fprintf('Starting the algorithm...\n');
for iteration = 1:itr
    for i = 1:m
        v = X(i,:) * theta;
        if v >= 0
            v = 1;
        else
            v = -1;
        end
        e(i) = t(i) - v;
        theta = theta + 2 * alpha * e(i) * X(i,:)';
    end
end

fprintf('\nAfter training with alpha = 0.0003, ');
fprintf('\nTheta found by training after %d iterations, which means calculated %d * %d times:\n', itr, itr, m);
fprintf('%.4f\n', theta);
fprintf('\n');
fprintf('The 2-norm of theta is %.2f\n', norm(theta));
fprintf('At this time, check if there is non-zero error in e: %d\n', any(e));
% =====================================================================

%% plot the data
data = sortrows(data,3);
m = -16:0.1:26;
n = (theta(1)/-theta(3)) + (theta(2)/-theta(3)) * m;
figure('Name','Original Data with Decision Boundary','NumberTitle','off');
scatter(data(1:1000,1), data(1:1000,2), 'o');
hold on;
scatter(data(1001:2000,1), data(1001:2000,2),'x');
hold on;
plot(m,n,'-')
legend('-1', '1', 'Decision boundary');
% =====================================================================

%% Validation
% Import test data
data_val = importdata('hmtest.mat');
x_val = data_val(:,1:2);
t_val = data_val(:,3);
[m_val,~] = size(x_val);

% Data processing
X_val = [ones(m_val,1) x_val(:,1:2)];
t_val(t_val == 0) = -1;

% Validation
predict = X_val * theta;
predict(predict >= 0) = 1;
predict(predict < 0) = -1;

accuracy = mean( double(predict == t_val) * 100);
fprintf('The accuracy is %.2f\n', accuracy);
% =====================================================================

%% Plot
data_val = sortrows(data_val,3);
figure('Name','Validation Data with Desicion Boundary','NumberTitle','off');
scatter(data_val(1:130,1), data_val(1:130,2), 'o');
hold on;
scatter(data_val(131:260,1), data_val(131:260,2),'x');
hold on;
plot(m,n,'-')
legend('-1', '1', 'Decision boundary');
% =====================================================================
