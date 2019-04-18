function J = computeCost(x, y, theta)
m = length(y);

h = logsig(x * theta);
J = (-1/m) * (y' * log(h + 0.01) + (1 - y)' * log(1 - h + 0.01));

end