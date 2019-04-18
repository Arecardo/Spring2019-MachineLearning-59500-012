function J = computeCost_part2(x, y, theta, lambda)

m = length(y);

h = logsig(x * theta);
J = (-1/m) * (y' * log(h) + (1 - y)' * log(1 - h)) + (lambda / (2 * m) ) * (theta' * theta);

end