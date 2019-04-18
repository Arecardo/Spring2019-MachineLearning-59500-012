function [theta_pyn, j_pyn_history] = gradientDescentPyn(X_pyn, y, theta_pyn, iteration_pyn, alpha_pyn)
m = 50;
j_pyn_history = zeros(iteration_pyn, 1);
lambda = 30; %* (1 - alpha_pyn * lambda / m)
% regularization is used here to get a better shape curve
x1 = X_pyn(:,2);
x2 = X_pyn(:,3);
x3 = X_pyn(:,4);

for i = 1:iteration_pyn
    hyp = theta_pyn(1) + theta_pyn(2)*x1 + theta_pyn(3)*x2 + theta_pyn(4)*x3;
    
    theta_zero = theta_pyn(1) - alpha_pyn * (1/m) * sum(hyp - y);
    theta_one = theta_pyn(2) - alpha_pyn * (1/m) * sum((hyp - y).*x1);
    theta_two = theta_pyn(3) * (1 - alpha_pyn * lambda / m) - alpha_pyn * (1/m) * sum((hyp - y).*x2); % regularized theta_two
    theta_three = theta_pyn(4) * (1 - alpha_pyn * lambda / m) - alpha_pyn * (1/m) * sum((hyp - y).*x3); % regularized theta_three

    theta_pyn = [theta_zero; theta_one; theta_two; theta_three];
    j_pyn_history(i) = computeCost(X_pyn, y, theta_pyn);
end
figure('Name','Polynomial Regression: J vs Iteration Index','NumberTitle','off');
plot(j_pyn_history); % plot J vs iteration index
ylabel('J');
xlabel('Iteration');
end