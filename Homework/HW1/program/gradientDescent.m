function [theta, J_history ] = gradientDescent(X, y, theta, iteration, alpha)

m = length(y);
J_history = zeros(iteration, 1);
x = X(:,2);

for i = 1:iteration
    hypothesis = theta(1) + (theta(2)*x); %prevent theta(1) from affecting by theta(2);
    
    theta_zero = theta(1) - (alpha / m) * sum(hypothesis - y);
    theta_one = theta(2) - (alpha /m)* sum((hypothesis - y) .*x);
    
    theta = [theta_zero; theta_one];
    J_history(i) = computeCost(X, y, theta);

end

figure('Name','Uni LR: J vs Iteration Index','NumberTitle','off');
plot(J_history); % plot J vs iteration index
ylabel('J');
xlabel('Iteration');
end