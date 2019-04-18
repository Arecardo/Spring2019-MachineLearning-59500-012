function J = computeCost(X, y, theta)

m = length(y);
J = 0;

hypothesis = X * theta;
squareError = (hypothesis - y).^2;
J = 1/ (2*m) * sum(squareError);
end