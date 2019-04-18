function x_nor = normalizing(x)
x_nor = x;
x_max = max(x);
x_min = min(x);
x_mean = mean(x);

for i = 1:50
            x_nor(i) = (x(i) - x_mean)/(x_max - x_min);
end