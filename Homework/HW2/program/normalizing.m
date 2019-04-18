function x_nor = normalizing(x)
x_nor = x;
%delta = x_max - x_min;
x_mean = mean(x);
x_nor = (x - x_mean)./ (std(x));

end
