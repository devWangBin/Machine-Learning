function [W] = MyLDA(w1,w2)
%W最大特征值对应的特征向量
%w1 第一类样本
%w2 第二类样本

%第一步：计算样本均值向量
m1=mean(w1);%第一类样本均值
m2=mean(w2);%第二类样本均值
m=mean([w1;w2]);%总样本均值

%第二步：计算类内离散度矩阵Sw
n1=size(w1,1);%第一类样本数
n2=size(w2,1);%第二类样本数
  %求第一类样本的散列矩阵s1
s1=0;
for i=1:n1
    s1=s1+(w1(i,:)-m1)'*(w1(i,:)-m1);
end
  %求第二类样本的散列矩阵s2
s2=0;
for i=1:n2
    s2=s2+(w2(i,:)-m2)'*(w2(i,:)-m2);
end
Sw=(n1*s1+n2*s2)/(n1+n2);
%第三步：计算类间离散度矩阵Sb
Sb=(n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);
%第四步：求最大特征值和特征向量
%[V,D]=eig(inv(Sw)*Sb);%特征向量V，特征值D
A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=eig(inv(Sw + B)*Sb);
[a,b]=max(max(D));
W=V(:,b);%最大特征值对应的特征向量
end