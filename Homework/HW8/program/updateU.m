function u = updateU(u, x, y, count, k, m)

for j = 1:k % for each category/label
    sum_c = zeros(1,2); % calculate the new centroids
    for i = 1:m
        if(y(i) == j)
            sum_c = sum_c + x(i,:);
        end
    end
    u(:,j) = (sum_c') / count(j);
end

end

