function x = normalization(x)

x_mean = mean(x);
x = (x - x_mean)./ (std(x));

end