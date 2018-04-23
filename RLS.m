clc
clear all

x = 1:200;
y = x.^2+0.5*x+1*randn(1,1);

%LS
matrix = [];
vector = [];
for i=1:length(x)
    matrix = [matrix; x(i)^2 x(i)];
    vector = [vector; y(i)];
end
coeffs_act = matrix\vector;

% %RLS
% coeffs = matrix(1:2,:)\vector(1:2);
% S = matrix(1:2,:)'*matrix(1:2,:);
% for i=3:length(x)
%     x_N = [x(i)^2 x(i)];
%     y_N = y(i);
%     S = S+x_N'*x_N;
%     coeffs = coeffs + S\(x_N'*(y_N - x_N*coeffs));
%     scatter(i,coeffs(1))
%     hold on
%     scatter(i,coeffs_act(1),'.')
%     hold on
%     pause(0.001)
% end

%GD
coeffs_GD = [0;0];
alpha = [];
for j=1:10
    gradient = zeros(2,1);
    for i=1:length(x)
        gradient = gradient+(matrix(i,:)*coeffs_GD-vector(i))*matrix(i,:)';
    end
    %Use minimization rule to find alpha--
    num = 0;
    den = 0;
    for i=1:length(x)
        num = num+ (vector(i)*matrix(i,:)*gradient-matrix(i,:)*coeffs_GD*matrix(i,:)*gradient);
        den = den + (matrix(i,:)*gradient)^2;
    end
    alpha = [alpha num/den];
    %--------------------------------------------
    %Take gradient step
    coeffs_GD = coeffs_GD+alpha(end)*gradient;
    %--------------------------------------------
    scatter(j,coeffs_GD(2))
    hold on
    scatter(j,coeffs_act(2),'.')
    hold on
    pause(0.001)
    coeffs_GD
end

%SGD
coeffs_SGD = [0;0];
alpha =[];
for j=1:1000
    for i=1:length(x)
        %Find a sample of the gradient
        gradient = (matrix(i,:)*coeffs_SGD-vector(i))*matrix(i,:)';
        %For that gradient, find minimizing alpha
        num = (vector(i)*matrix(i,:)*gradient-matrix(i,:)*coeffs_SGD*matrix(i,:)*gradient);
        den = (matrix(i,:)*gradient)^2;
        alpha = [alpha num/den];
        %Use that alpha for taking a step
        coeffs_SGD = coeffs_SGD+alpha(end)*gradient;
        scatter(i,coeffs_SGD(1))
        hold on
        scatter(i,coeffs_act(1),'.')
        hold on
        pause(0.001)
        alpha(end)
    end
    coeffs_SGD
end
