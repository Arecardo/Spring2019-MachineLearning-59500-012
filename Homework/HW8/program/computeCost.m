function J = computeCost(u, x, y, k, m)
J = 0;
for j = 1:k
    for i = 1:m
        if(y(i) == j)
            J = J + norm(u(:,j) - x(i,:)');
        end
    end
end

end

