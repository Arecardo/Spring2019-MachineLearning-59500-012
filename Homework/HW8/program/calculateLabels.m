function [y, count] = calculateLabels(u, x, y, k, m)
d = zeros(1,5);
count = zeros(5,1); % Initial matrix count to store the number of each labels
for i = 1:m % for each data point
    v = u - x(i,:)'; % calculate the distance from data point to each centroid
    for j = 1:k 
        d(j) = sqrt(v(:,j)'*v(:,j));
    end
    index = find(d == min(d)); % find the min distance
    y(i) = index; % add the label 
    count(index)  = count(index) + 1; % count the number of this label
end

end
