function theta = gradientDescent(x, y, theta, iteration, alpha)

m = length(y);

for i = 1:iteration
    h = logsig(x * theta);
    theta = theta - (alpha / m) * (x' * (h - y));
end

end