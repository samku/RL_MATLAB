clc
clear all

%System dynamics
a1 = 0.9;
b1 = 1.0;
a2 = 0.9;
b2 = b1;
p = 1.0;

%Parameters
init_state = 0;
final_state = 10;    
N_steps = 50;
N_episodes = 100;
gamma = 1.0;

%Cost weights
Q = 1;
R = 1;
[K,~,~] = dlqr(a1,b1,Q,R,0);
x_sim = init_state;
for i=2:N_steps
    x_sim = [x_sim a1*x_sim(i-1)+b1*K*(final_state-x_sim(i-1))];
end

%Q function parameterization
% q_xu = @(x,u,theta)[x^2 x u^2 u x*u]*theta;
% feature_xu = @(x,u)[x^2;x;u^2;u;x*u];
% u_opt = @(x,theta)(-(theta(4)+x*theta(5))/(2*theta(3)));
feature_xu = @(x,u)[(10-x)^2;(10-x)*u;u^2];
q_xu = @(x,u,theta)feature_xu(x,u)'*theta;
u_opt = @(x,theta)(-theta(2)/(2*theta(3)))*(10-x);
theta = ones(3,1);

type = 2;
alpha = [];
if type == 1
    %Simulation - Monte carlo value gradient
    cost_MC = [];
    K_vec =[];
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %Apply to dynamics - Observe next state
             u(i,j-1) = u_opt(x(i,j-1),theta)+(0.5/i)*randn(1,1);
%             K_vec = [ K_vec;(H(2,1)/H(2,2))];
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
        end
% --------------------------------------------------------------------
%Learning phase
        %Use every visit MC
        %Calculate cost targets first
        for j=1:N_steps-1
                net_cost(j) = 0;
                for k=j:N_steps
                    error_k = final_state - x(i,k);
                    net_cost(j) = net_cost(j) + gamma^(k-j)*(Q*error_k^2+R*u(i,k)^2);
                end
        end
        %Use complete gradient descent
        for gd_iter = 1:10
            %Calculate gradient at evaluated point
            dtheta_sum = zeros(size(theta));
            for j=1:N_steps-1
                %Enumerate gradient
                dtheta_sum = dtheta_sum+(q_xu(x(i,j),u(i,j),theta)-net_cost(j))*feature_xu(x(i,j),u(i,j));
            end
            %Calculate optimal step size at the current point
            %Use minimization rule to find alpha--
            num = 0;
            den = 0;
            for j=1:N_steps-1
                num = num+ (net_cost(j)-feature_xu(x(i,j),u(i,j))'*theta);
                den = den + (feature_xu(x(i,j),u(i,j))'*dtheta_sum);
            end
            alpha = [alpha num/den];
            %Take a gradient step
            theta = theta+alpha(end)*dtheta_sum;
            subplot(2,3,1)
            scatter(gd_iter,theta(1),'.')
            hold on
            subplot(2,3,2)
            scatter(gd_iter,theta(2),'.')
            hold on
            subplot(2,3,4)
            scatter(gd_iter,theta(3),'.')
            hold on
%             scatter(gd_iter,theta(4),'.')
            hold on
            pause(0.001)
        end
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_MC = [cost_MC cost];
        %Check cost
        subplot(2,3,6)
        scatter(i,cost)
        hold on
        subplot(2,3,5)
        plot(x(i,:))
        hold on
        pause(0.0001)
    end
end

type = 2;
alpha = [];
if type == 2
    %Simulation - Q -learning
    cost_MC = [];
    K_vec =[];
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %Apply to dynamics - Observe next state
             u(i,j-1) = u_opt(x(i,j-1),theta)+(0.1)*randn(1,1);
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
            %Evaluate gradient and modify Q
            u_next = u_opt(x(i,j),theta);
            target = Q*(final_state-x(i,j-1))^2+R*u(i,j-1)^2+gamma*q_xu(x(i,j),u_next,theta);
            gradient = (q_xu(x(i,j-1),u(i,j-1),theta)-target)*feature_xu(x(i,j-1),u(i,j-1));
            %Get optimal step length
            alpha = [alpha (target-q_xu(x(i,j-1),u(i,j-1),theta))/(feature_xu(x(i,j-1),u(i,j-1))'*gradient)];
            %Update theta
            theta = theta+alpha(end)*gradient;
            subplot(2,3,1)
            scatter(j,theta(1),'.')
            hold on
            subplot(2,3,2)
            scatter(j,theta(2),'.')
            hold on
            subplot(2,3,4)
            scatter(j,theta(3),'.')
            hold on
            pause(0.001)
        end
% -------------------------------------------------------------------- 
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_MC = [cost_MC cost];
        %Check cost
        subplot(2,3,6)
        scatter(i,cost)
        hold on
        subplot(2,3,5)
        plot(x(i,:))
        hold on
        pause(0.0001)
    end
end