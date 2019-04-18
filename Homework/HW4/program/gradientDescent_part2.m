function theta = gradientDescent_part2(x, y, theta, iteration, alpha, lambda)

m = length(y);

for i = 1:iteration
    h = logsig(x * theta);
    foo = [0; theta(2:28, 1)];
    theta = theta - (alpha / m) * (x' * (h - y)) + (lambda / m) .* foo ;
end

end