clc
clear all

%System dynamics
a1 = 0.6;
b1 = 1.0;
a2 = a1;
b2 = b1;
p = 1.;

%Parameters
init_state = 10;
final_state = 0;
N_steps = 1000;
N_episodes = 1000;
gamma = 1.0;
alpha = -0.01;

%Cost weights
Q = 1;
R = 1;
[K,~,~] = dlqr(a1,b1,Q,R,0);
x_sim = init_state;
cost_opt = 0;
for i=2:N_steps
    cost_opt = cost_opt+Q*x_sim(i-1)^2 + R*K*x_sim(i-1)^2;
    x_sim = [x_sim a1*x_sim(i-1)+b1*K*(final_state-x_sim(i-1))];
end
[K,~,~] = dlqr(a2,b2,Q,R,0);

%Check effect of finite horizon
P = 0;
P_vec = P;
for i=1:N_steps
    P = Q+a1*P*a1 - (a1*P*b1*(R+(b1*P*b1))^(-1)*b1*P*a1);
    P_vec = [P_vec; P];
end

%State distrubution
x_step = 0.001;
X = round(-init_state*10:x_step:init_state*10,log10(1/x_step));
counter_table = zeros(1,length(X));

%Functions
feature = @(x)[0*x^2;x;0*1];
policy = @(x,a,theta)[exp(-1*(a-feature(x)'*theta)^2)];

%Initializtion
theta = ones(3,1)/100;
theta(1) = 0;
theta(2) = -0.5;
theta(3) = 0;

%REINFORCE algorithm
%Blindly apply and see
%Simulation - Monte carlo policy gradient
cost_MC = [];
K_vec =[];
x = zeros(N_episodes,N_steps);
u = zeros(N_episodes,N_steps);
for i=1:N_episodes
    %Generate an episode
    x(i,1) = init_state;
    for j=2:N_steps
        %Apply to dynamics - Observe next state
        u(i,j-1) = feature(x(i,j-1))'*theta+0.1*randn(1);
        if x(i,j-1)<=5
            x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
        else
            x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
        end
    end
    % --------------------------------------------------------------------
    %Learning phase
    if 1==1
        %Use every visit MC
        %Calculate cost targets first
        %theta_prev = theta;
         for num_iters = 1:5
            counter_table = zeros(1,length(X));
            gradient = zeros(3,1);
            for j=1:N_steps-1
                net_cost(j) = 0;
                for k=j:N_steps
                    error_k = final_state - x(i,k);
                    net_cost(j) = net_cost(j) + gamma^(k-j)*(Q*error_k^2+R*u(i,k)^2);
                end
                %Calculation state distrubution
                x_location(j) = find(X == round(x(i,j),log10(1/x_step)));
                counter_table(1,x_location(j)) = counter_table(1,x_location(j))+1;
            end
            %Normalize distrbution
            counter_table = counter_table/sum(counter_table);
            counter_table = ones(size(counter_table));
            %Calculate gradient
            for j=1:N_steps-1
                gradient = gradient+counter_table(1,x_location(j))*exp(-1*(u(i,j)-feature(x(i,j))'*theta)^2)*(2*(u(i,j)-feature(x(i,j))'*theta))*feature(x(i,j))*net_cost(j);
            end
%             %Use backtracking to find alpha
            alpha = 0.00001;
            num_times = 0;
            beta = 0.5;
            sigma = 0.1;
            condition_met = 0;
            old_value = 0;
            new_value = 0;
            while condition_met == 1
                num_times = num_times+1;
                for j=1:N_steps-1
                    old_value = old_value+counter_table(1,x_location(j))*exp(-1*(u(i,j)-feature(x(i,j))'*theta)^2)*net_cost(j);
                    new_value = new_value+counter_table(1,x_location(j))*exp(-1*(u(i,j)-feature(x(i,j))'*(theta+alpha*gradient))^2)*net_cost(j);
                end
                if new_value<=old_value || num_times>=100
                    condition_met = 1;
                else
                    alpha=alpha*beta;
                end
            end
            %Update theta
            theta = theta+(alpha)*gradient;
         end
         %theta = theta_prev+(1/(i+1))*(theta-theta_prev);
    else
        theta_prev = theta;
        theta = sdpvar(3,1);
        f = 0;
        counter_table = zeros(1,length(X));
        for j=1:N_steps-1
                net_cost(j) = 0;
                for k=j:N_steps
                    error_k = final_state - x(i,k);
                    net_cost(j) = net_cost(j) + gamma^(k-j)*(Q*error_k^2+R*u(i,k)^2);
                end
                x_location(j) = find(X == round(x(i,j),log10(1/x_step)));
                counter_table(1,x_location(j)) = counter_table(1,x_location(j))+1;
        end
        counter_table = counter_table/sum(counter_table);
        for j=1:N_steps-1
            if counter_table(1,x_location(j))>=0
                f = f+counter_table(1,x_location(j))*(exp(-1*(u(i,j)-feature(x(i,j))'*theta)^2)*net_cost(j));
            end
        end
        options = sdpsettings('verbose',10,'solver','fmincon','fmincon.MaxIter',20);
        optimize([theta(1)==0;theta(3)==0],f,options);
        value(theta)
        theta = theta_prev+(1/(i+1))*(value(theta)-theta_prev);
    end
    %     scatter(i,theta(1),'.')
    %           hold on
    %           scatter(i,theta(2),'.')
    %           hold on
    %           scatter(i,theta(3),'.')
    %           hold on
    %           pause(0.001)
    %Check gain
        subplot(1,3,1)
    K_vec = [K_vec -feature(10)'*theta/10];
    K_vec(end)
    scatter(i,K_vec(end),'.','black')
    hold on
    scatter(i,K,'.')
    hold on
    pause(0.001)
    %-----------------------------------------------------------------------
    %Check if performance is improving
    cost  =0;
    for j=1:N_steps
        cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
    end
    cost_MC = [cost_MC cost];
    %Check cost
    subplot(1,3,2)
        scatter(i,cost,'.','black')
        hold on
    scatter(i,cost_opt,'.')
        hold on
        pause(0.0001)
    
    x_surf = 0:0.1:10;
    a_surf = -5:0.1:5;
    clear pi_surf
    for i=1:length(x_surf)
        for j=1:length(a_surf)
            pi_surf(i,j) = policy(x_surf(i),a_surf(j),theta);
        end
    end
    subplot(1,3,3)
    [x_grid,a_grid] = meshgrid(x_surf,a_surf);
     surf(a_grid,x_grid,pi_surf')
     pause(0.001)
end

