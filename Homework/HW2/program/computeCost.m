function J = computeCost(X, y, theta)
m = length(y);
J = 0;

hypothesis = X * theta;
J = (1/ (2*m)) * (hypothesis - y)' * (hypothesis - y);
end