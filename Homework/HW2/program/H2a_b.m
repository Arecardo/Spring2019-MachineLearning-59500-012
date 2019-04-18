clc;
clear;

%H2_a
A = xlsread('D:\我的文档\桌面\AutoData_HW2.xlsx');

% Read the input data from column 2 to 5 with output data column 1
x = A(1:160,2:5);
y = A(1:160,1);

%Normalion the input data
X(:,1) = (x(:,1) - mean(x(:,1))) ./ std(x(:,1));
X(:,2) = (x(:,2) - mean(x(:,2))) ./ std(x(:,2));
X(:,3) = (x(:,3) - mean(x(:,3))) ./ std(x(:,3));
X(:,4) = (x(:,4) - mean(x(:,4))) ./ std(x(:,4));

m = length(y);
X = [ones(m,1),X];  % Add a column of ones to X; 

theta = zeros(5,1); %Initialize the theta
alpha = 0.1;
iterations = 5000;

% Cost function 
J = computeCostMulti(X, y, theta);

% Gradient Descent function
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);
fprintf('theta:%f\n',theta);


%H2_b
%Add the input data from row 161 to 200
x_p = A(161:200,2:5);

x_p(:,1) = (x_p(:,1) - mean(x_p(:,1))) ./ std(x_p(:,1)); %Normalize
x_p(:,2) = (x_p(:,2) - mean(x_p(:,2))) ./ std(x_p(:,2));
x_p(:,3) = (x_p(:,3) - mean(x_p(:,3))) ./ std(x_p(:,3));
x_p(:,4) = (x_p(:,4) - mean(x_p(:,4))) ./ std(x_p(:,4));


X_p = [ones(40,1),x_p];  % Add a column of ones to predict x_p; 

y_p = X_p*theta; % Get the true output data in the excel file
y_true = A(161:200,1); % Get the true output data in the excel file

err_cv = (1/40)*sum((y_p-y_true).^2); % Do the MSE calculation
fprintf('mes:%f',err_cv); 
