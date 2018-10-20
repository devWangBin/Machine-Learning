function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
%m=47
J_history = zeros(num_iters, 1);
n = size(X , 2);
%n=3
for iter = 1:num_iters
    H = X * theta;
    %(47,3)*(3*1) = (47*1)
    T = zeros(n , 1);
    %3*1
    for i = 1 : m,
        T = T + (H(i) - y(i)) * X(i,:)';    
    end
    theta = theta - (alpha * T) / m;
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end
end