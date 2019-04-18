function [ J, grad ] = computeCost_new(theta, x, y)
m = length(y);

h = logsig(x * theta);
J = (-1/m) * sum(y .* log(h) + (1 - y) .* log(1 - h));

grad = (1 / m) * ( (h - y)' * x );

end