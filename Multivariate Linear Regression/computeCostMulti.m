function J = computeCostMulti(x,y,theta)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明

m=length(y);
J = 0;
% 初始化
J = sum((x*theta - y).^2) / (2 * m);
% 计算损失


end

