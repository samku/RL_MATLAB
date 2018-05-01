clc
clear all

%System dynamics
a1 = 0.9;
b1 = 1.0;
a2 = 0.9;
b2 = 1.0;
p = 1.0;

%Parameters
init_state = 0;
final_state = 10;
x_step = 0.1;
u_min = -1;
u_max = 1;
u_step = 0.01;
N_steps = 1000;
N_episodes = 2000;
gamma = 1.0;
epsilon = 0.2;
factor = 1.0;

%Cost weights
Q = 1;
R = 0.;

%Spaces
A = round(u_min:u_step:u_max,log10(1/u_step));
X = round(-final_state*2:x_step:final_state*2,log10(1/x_step));

%Tables
Q_table = rand(length(A),length(X));
pi_table = rand(length(A),length(X));
for i=1:length(X)
    pi_table(:,i) = pi_table(:,i)/sum(pi_table(:,i));
end
returns_table = zeros(length(A),length(X));
counter_table = zeros(length(A),length(X));
%Save for future
Q_orig = Q_table;
pi_orig = pi_table;
returns_orig = returns_table;
counter_orig = counter_table;

%Type
type = 1;
%1 - MC, 2- SARSA, 3-Q-learning

if type == 1
    %Simulation - Monte carlo
    cost_MC = [];
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        i
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %Sample an actions from current state
            action_id = rejection_sampling(pi_table(:,find(X == round(x(i,j-1),log10(1/x_step)))));
            %Apply to dynamics - Observe next state
            u(i,j-1) = A(action_id);
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
        end
        %Use every visit MC
        for j=1:N_steps-1
            x_location = find(X == round(x(i,j),log10(1/x_step)));
            u_location = find(A == round(u(i,j),log10(1/u_step)));
            cc
            %Calculate total cost till end
            net_cost = 0;
            for k=j:N_steps
                net_cost = net_cost + gamma^(k-j)*(Q*(final_state-x(i,k))^2+R*u(i,k)^2);
            end
            Q_table(u_location,x_location) = Q_table(u_location,x_location)+(1/counter_table(u_location,x_location))*(net_cost-Q_table(u_location,x_location));
        end
        %Update policy for each s in the episode
        for j=1:N_steps
            x_location = find(X == round(x(i,j),log10(1/x_step)));
            [temp,min_id] = min(Q_table(:,x_location));
            pi_table(:,x_location) =  epsilon/i^(factor);
            pi_table(min_id,x_location) = 1-epsilon/i^(factor);
            pi_table(:,x_location) = pi_table(:,x_location)/sum(pi_table(:,x_location));
        end
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_MC = [cost_MC cost];
        %Check cost
%         scatter(i,cost)
%         hold on
%         pause(0.0001)
    end
end

figure
subplot(1,2,1)
plot(x(i,:))
hold on
subplot(1,2,2)
plot(cost_MC)
hold on
pause(0.001)

keyboard


[a_grid,x_grid] = meshgrid(A,X);
surf(a_grid,x_grid,pi_table')

%Sarsa
type = 2
Q_table = Q_orig;
pi_table = pi_orig;
returns_table = returns_orig;
counter_table = counter_orig;

if type == 2
    cost_sarsa = [];
    %Learning rate
    alpha = 0.5;
    %Simulation - Sarsa
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        i
        %Generate an episode
        x(i,1) = init_state;
        %--------------------------------
        %Choose an action epsilon-greedily
        % For current state, based on Q, update pi first. Then sample
        % action
        x_location = find(X == round(x(i,1),log10(1/x_step)));
        [temp,min_id] = min(Q_table(:,x_location));
        pi_table(:,x_location) =  epsilon/i^(factor);
        pi_table(min_id,x_location) = 1-epsilon/i^(factor);
        pi_table(:,x_location) = pi_table(:,x_location)/sum(pi_table(:,x_location));
        action_id = rejection_sampling(pi_table(:,x_location));
        %-------------------------------
        for j=2:N_steps
            %Apply to dynamics - Observe next state
            u(i,j-1) = A(action_id);
              if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
            %--------------------------------
            %Choose an action epsilon-greedily
            % For current state, based on Q, update pi first. Then sample
            % action
            x_location = find(X == round(x(i,j),log10(1/x_step)));
            [temp,min_id] = min(Q_table(:,x_location));
            pi_table(:,x_location) =  epsilon/i^(factor);
            pi_table(min_id,x_location) = 1-epsilon/i^(factor);
            pi_table(:,x_location) = pi_table(:,x_location)/sum(pi_table(:,x_location));
            action_dash_id = rejection_sampling(pi_table(:,x_location));
            %-------------------------------
            %Update Q value of (j-1) state
            x_location = find(X == round(x(i,j-1),log10(1/x_step)));
            x_dash_location = find(X == round(x(i,j),log10(1/x_step)));
            Q_sa = Q_table(action_id,x_location);
            Q_sa_dash = Q_table(action_dash_id,x_dash_location);
            r = Q*(final_state-x(i,j-1))^2+R*u(i,j-1)^2;
            Q_sa = Q_sa+alpha/(i^factor)*(r+gamma*Q_sa_dash-Q_sa);
            Q_table(action_id,x_location) = Q_sa;
            %Take same sampled action at next time step
            action_id = action_dash_id;
        end
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_sarsa = [cost_sarsa cost];
        %Check cost
%         scatter(i,cost)
%         hold on
%         pause(0.0001)
    end
end
subplot(1,2,1)
plot(x(i,:))
hold on
subplot(1,2,2)
plot(cost_sarsa)
hold on
pause(0.001)

%Q learning
type = 3
Q_table = Q_orig;
pi_table = pi_orig;
returns_table = returns_orig;
counter_table = counter_orig;

if type == 3
    cost_Q_learning = [];
    %Learning rate
    alpha = 0.5;
    %Simulation - Q-learning
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        i
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %--------------------------------
            %Choose an action epsilon-greedily
            % For current state, based on Q, update pi first. Then sample
            % action
            x_location = find(X == round(x(i,j-1),log10(1/x_step)));
            [temp,min_id] = min(Q_table(:,x_location));
            pi_table(:,x_location) =  epsilon/i^(factor);
            pi_table(min_id,x_location) = 1-epsilon/i^(factor);
            pi_table(:,x_location) = pi_table(:,x_location)/sum(pi_table(:,x_location));
            action_id = rejection_sampling(pi_table(:,x_location));
            %-------------------------------s
            %Apply to dynamics - Observe next state
            u(i,j-1) = A(action_id);
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
            %--------------------------------
            %Choose max value action for next state
            x_location = find(X == round(x(i,j),log10(1/x_step)));
            [temp,action_dash_id] = min(Q_table(:,x_location)); 
            %-------------------------------
            %Update Q value of (j-1) state
            x_location = find(X == round(x(i,j-1),log10(1/x_step)));
            x_dash_location = find(X == round(x(i,j),log10(1/x_step)));
            Q_sa = Q_table(action_id,x_location);
            Q_sa_dash = Q_table(action_dash_id,x_dash_location);
            r = Q*(final_state-x(i,j-1))^2+R*u(i,j-1)^2;
            Q_sa = Q_sa+(alpha/i^factor)*(r+gamma*Q_sa_dash-Q_sa);
            Q_table(action_id,x_location) = Q_sa;
            %-------------------------------
        end
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_Q_learning = [cost_Q_learning cost];
        %Check cost
%         scatter(i,cost)
%         hold on
%         pause(0.0001)
    end
end
subplot(1,2,1)
plot(x(i,:))
hold on
grid on
legend('MC','Sarsa','Q-learning')
subplot(1,2,2)
plot(cost_Q_learning)
grid on
legend('MC','Sarsa','Q-learning')
