function ex5Lin(lambda)
% 正则化线性回归
%加载数据
close all; clc
x = load('ex5Linx.dat'); y = load('ex5Liny.dat');

str1 = 'lambda = ';
str2 = num2str(lambda);
Title = [str1,str2];

m = length(y); % 训练样本数量

% 样本展示
figure;
plot(x, y, 'o', 'MarkerFacecolor', 'b', 'MarkerSize', 8);

% 构建从x的0次方到x的5次方的样本数据矩阵并初始化参数theta
x = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];
theta = zeros(size(x(1,:)))';

% 定义正则化超参
la = lambda;
L = la.*eye(6); 
L(1) = 0;
theta = (x' * x + L)\x' * y
norm_theta = norm(theta)


hold on;
% 构建密集的矩阵来将拟合的曲线显示出来
x_vals = (-1:0.05:1)';
features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3,...
          x_vals.^4, x_vals.^5];
plot(x_vals, features*theta, '--', 'LineWidth', 2)
title(Title)
legend('样本点', '拟合曲线')
hold off


