function [X_pyn, x_predict] = featureScaling(x_pyn, x_predict)
X_pyn = x_pyn;
Sd = std(x_pyn); % compute the standard deviation of each column;
me = mean(x_pyn); % compute the mean value of each column;

for i = 1:3
        X_pyn(:, i) = (x_pyn(:, i) - me(i)) / Sd(i);
end

x_predict = (x_predict - me(1)) / Sd(1);

end