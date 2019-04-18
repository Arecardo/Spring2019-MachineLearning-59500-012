function theta = trainingNueron( theta, p, t, alpha )
iter = size(t);
for i = 1:iter    
    h = round(logsig(p(i, :)*theta));
    error = t(i) - h;
    theta = theta + alpha * p(i, :)' * error;
end

end
