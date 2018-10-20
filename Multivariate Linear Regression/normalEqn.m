function [theta] = normalEqn(X, y)
theta = zeros(size(X, 2), 1);
theta = pinv(X' * X) * X' * y;

%pinv(a)是求伪逆矩阵，逆矩阵函数inv只能对方阵求逆，pinv(a)可以对非方阵求逆。
end