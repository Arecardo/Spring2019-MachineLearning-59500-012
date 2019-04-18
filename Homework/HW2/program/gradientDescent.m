function [theta, J_history ] = gradientDescent(X, y, theta, iteration, alpha)

m = length(y);
J_history = zeros(iteration, 1);

for i = 1:iteration
    hypothesis = X*theta;
    theta = theta - (alpha / m) * (X' * (hypothesis - y));
    J_history(i) = computeCost(X, y, theta);

end

figure('Name','Multi LR: J vs Iteration Index','NumberTitle','off');
plot(J_history); % plot J vs iteration index
ylabel('J');
xlabel('Iteration');
end