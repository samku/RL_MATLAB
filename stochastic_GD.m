clc
clear all

%Regression data
x = 1:5:100;
y = (x + 5*x.^2)/100;

z_func = -10:0.01:10;
for i=1:length(z_func)
    cost(i) = 0;
    for j=1:length(x)
        cost(i) = cost(i) + (x(j)*z_func(i) - y(j))^2;
    end
end

figure
subplot(2,2,1)
plot(z_func,cost)
xlim([min(z_func),max(z_func)]);
ylim([min(cost),max(cost)]);
grid on
hold on
subplot(2,2,3)
scatter(x,y)
grid on
hold on

%Coefficients
z=-10;
alpha = 1e-6;
iters = 100;
subplot(2,2,1)
scatter(z,norm(x'*z-y',2)^2)
hold on

%Usual gradient descent
for j=1:iters
    dz = 0;
    y1 = norm(x'*z-y',2)^2;
    for i=1:length(x)
        dz = dz+2*(x(i)*z-y(i))*x(i);
    end
    subplot(2,2,1)
    plot(z_func,y1+dz*(z_func-z),':');
    z = z-alpha*dz;
    hold on
    subplot(2,2,1)
    scatter(z,norm(x'*z-y',2)^2)
    hold on
    subplot(2,2,3)
    plot(x,x*z,':');
    hold on
end
z

%Stochastic gradient descent
subplot(2,2,2)
plot(z_func,cost)
xlim([min(z_func),max(z_func)]);
ylim([min(cost),max(cost)]);
grid on
hold on
subplot(2,2,4)
scatter(x,y)
grid on
hold on

%Coefficients
z=-10;
alpha = 1e-5;
iters = 10;
subplot(2,2,2)
scatter(z,norm(x'*z-y',2)^2)
hold on

%Usual gradient descent
for j=1:iters
    dz = 0;
    y1 = norm(x'*z-y',2)^2;
    for i=1:length(x)
        dz = dz+2*(x(i)*z-y(i))*x(i);
        subplot(2,2,2)
        plot(z_func,y1+dz*(z_func-z),':');
        z = z-alpha*dz;
        hold on
        subplot(2,2,2)
        scatter(z,norm(x'*z-y',2)^2)
        hold on
        subplot(2,2,4)
        plot(x,x*z,':');
        hold on
        pause
    end
end
z



