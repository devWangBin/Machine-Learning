function [X_norm,mu,sigma] = featureNormalize(X)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
X_norm = X;
%(47,2)
mu = zeros(1, size(X, 2));
%size(X, 2)取X的第二维，mu大小为（1,2）
sigma = zeros(1, size(X, 2));
%(1,2)
m = size(X,1);
%47
mu = mean(X);
%计算均值
for i = 1 : m,
    X_norm(i, :) = X(i , :) - mu;
end
sigma = std(X);
for i = 1 : m,
    X_norm(i, :) = X_norm(i, :) ./ sigma;
end
end

