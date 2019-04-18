%% Machine Learning Homework 8 part 2
% K-means algorithm with random centroids
% Author: Xinrun Zhang
% Time: 04/15/2019 16:00
% =====================================================================

%% Initializing
clear ; close all; clc
fprintf('Initializing...\n');
% Import the data
x = importdata('HW8.mat');
x = x(randperm(200),:);
[m, ~] = size(x);

% Initial k
% Initial random centroids
k = 5;
u = [x(1:5,1)';x(1:5,2)'];
fprintf('The random centroids are:\n');
disp(u);

% Initial matrix y to store labels
% Initial cost J
y = zeros(200,1);
J = 1;
% =====================================================================

%% K-means algorithm with random centroids u
% start the algorithm
fprintf('Start the K-means algorithm with initial centroids u...\n');
itr = 0;
% initial a cell to store centroids
u_store = cell(20,1);
while(1)
    itr = itr + 1;
    % store the centroids
    u_store{itr} = u';
    u_old = u;
    [y, count] = calculateLabels(u, x, y, k, m);
    u = updateU(u, x, y, count, k, m);
    % compute the cost J
    J = computeCost(u, x, y, k, m);
    if(norm(u_old - u) <= 0.0001)
        break;
    end
end
fprintf('After %d iterations, the u found by K-means algorithm:\n', itr);
disp(u);
fprintf('The final cost is %.4f\n', J);
% =====================================================================

%% plot
c1_history = zeros(itr, 2);
c2_history = zeros(itr, 2);
c3_history = zeros(itr, 2);
c4_history = zeros(itr, 2);
c5_history = zeros(itr, 2);
figure('Name','Data and centroids','NumberTitle','off');
scatter(x(:,1), x(:,2),'o');
hold on;
for i = 1:20
    if(~isempty(u_store{i}))
        c1_history(i,:) = u_store{i}(1,:);
        c2_history(i,:) = u_store{i}(2,:);
        c3_history(i,:) = u_store{i}(3,:);
        c4_history(i,:) = u_store{i}(4,:);
        c5_history(i,:) = u_store{i}(5,:);
    end
end
plot(c1_history(:,1),c1_history(:,2),'-*');
hold on;
plot(c2_history(:,1),c2_history(:,2),'-*');
hold on;
plot(c3_history(:,1),c3_history(:,2),'-*');
hold on;
plot(c4_history(:,1),c4_history(:,2),'-*');
hold on;
plot(c5_history(:,1),c5_history(:,2),'-*');
hold on;
legend('data', 'cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5')
hold off;