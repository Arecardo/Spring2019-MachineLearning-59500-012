%% Machine Learning Homework 8 part 1
% K-means algorithm with initial centroids
% Author: Xinrun Zhang
% Time: 04/14/2019 20:22
% =====================================================================

%% Initializing
clear ; close all; clc
fprintf('Initializing...\n');

% Import the data
x = importdata('HW8.mat');
[m, ~] = size(x);

% Initial k and centroid u matrix
k = 5;
u = [20, 20, 60, 85, 80; 15, 80, 50, 20, 90];

% Initial D
D = 2603.5;

% Initial matrix y to store labels
% Initial cost J
y = zeros(200,1);
J = 1;

% =====================================================================

%% K-means algorithm with initial centroids u
% start the algorithm
fprintf('Start the K-means algorithm with initial centroids u...\n');
itr = 0;
J_history = zeros(3,1);
while(1)
    [y, count] = calculateLabels(u, x, y, k, m);
    u = updateU(u, x, y, count, k, m);
    % compute the cost J
    itr = itr + 1;
    J_history(itr) = computeCost(u, x, y, k, m);
    if(J_history(itr) < D)
        break;
    end
end
fprintf('After %d iterations, the u found by K-means algorithm:\n', itr);
disp(u);
fprintf('The final cost is %.4f\n', J_history(itr));
% =====================================================================

%% Plot
figure('Name','Data and centroids','NumberTitle','off');
scatter(x(:,1), x(:,2),'x');
hold on;
scatter(u(1,:), u(2,:), 'o');
legend('data', 'centroids')

figure('Name','K-means: J vs Iteration Index','NumberTitle','off');
plot(J_history); % plot J vs iteration index
ylabel('J');
xlabel('Iteration');