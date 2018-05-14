clc
clear all

%System dynamics
a1 = 0.9;
b1 = 1.0;
a2 = a1;
b2 = b1;
p = 1.0;

%Parameters
init_state = 20;
final_state = 0;    
N_steps = 10;
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

%Initializtion
H = ones(2,2);

%Type
type = 3;
%1 - MC, 2- MC, RLS

if type == 1                             
    %Simulation - Monte carlo - RLS
    cost_MC = [];
    K_vec =[];   
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        i
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %Apply to dynamics - Observe next state
            u(i,j-1) = (H(2,1)/H(2,2))*(final_state-x(i,j-1))+(0.01)*randn(1,1);
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
        end
        %Use every visit MC
        matrix = [];
        vector = [];
        for j=1:N_steps-1
            x_i = x(i,j);
            u_i = u(i,j);
            x_n =x(i,j+1);
            u_n = u(i,j+1);
            if j<=3
                matrix = [matrix; x_i^2-gamma*x_n^2  2*(x_i*u_i-gamma*x_n*u_n) u_i^2-gamma*u_n^2];
                vector = [vector; (final_state-x_i)^2*Q + u_i^2*R];
            end
            if j==3
                %Solve for 4 first-------
                H_elements = matrix\vector;
                S = matrix'*matrix;
                %--------------------------
            end
            if j>3
                X_N = [x_i^2 - gamma*x_n^2 2*(x_i*u_i-gamma*x_n*u_n) u_i^2 - gamma*u_n^2];
                Y_N = x_i^2*Q + u_i^2*R;
                S = S+X_N'*X_N;
                H_elements = H_elements + S\(X_N'*(Y_N - X_N*H_elements));
            end
        end
        H = [H_elements(1) H_elements(2); H_elements(2) H_elements(3)];
        K_vec = [ K_vec;(H(2,1)/H(2,2))];
        K_vec(end);
        scatter(i,K_vec(end))
        hold on
        scatter(i,K,'.')
        hold on
        pause(0.001)
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_MC = [cost_MC cost];
    end
end

if type == 2
    %Simulation - Q_learning RLS
    cost_MC = [];
    K_vec =zeros(N_episodes,N_steps);
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    for i=1:N_episodes
        i
        matrix = [];
        vector = [];
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %Apply to dynamics - Observe next state
            u(i,j-1) = (H(2,1)/H(2,2))*(final_state-x(i,j-1))+(0.01)*randn(1,1);
            if x(i,j-1)<=10
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
            %Compute H
            x_i = x(i,j-1);
            u_i = u(i,j-1);
            x_n =x(i,j);
            u_n = (H(2,1)/H(2,2))*(final_state-x_n);
            if j<=4
                matrix = [matrix; x_i^2-gamma*x_n^2  2*(x_i*u_i-gamma*x_n*u_n) u_i^2-gamma*u_n^2];
                vector = [vector; (final_state-x_i)^2*Q + u_i^2*R];
                j
            end
            if j==4
                %Solve for 4 first-------
                H_elements = matrix\vector;
                S = matrix'*matrix;
                H = [H_elements(1) H_elements(2); H_elements(2) H_elements(3)];
                %--------------------------
            end
            if j>4
                X_N = [x_i^2 - gamma*x_n^2 2*(x_i*u_i-gamma*x_n*u_n) u_i^2 - gamma*u_n^2];
                Y_N = x_i^2*Q + u_i^2*R;
                S = S+X_N'*X_N;
                H_elements = H_elements + S\(X_N'*(Y_N - X_N*H_elements));
                H = [H_elements(1) H_elements(2); H_elements(2) H_elements(3)];
            end
            K_vec(i,j) = (H(2,1)/H(2,2));
        end
        plot(1:N_steps,K_vec(i,:))
        hold on
        scatter(i,K,'.')
        hold on
        pause(0.001)
        %Compute LQ cost
        cost  =0;
        for j=1:N_steps
            cost = cost+Q*(final_state-x(i,j))^2 + R*u(i,j)^2;
        end
        cost_MC = [cost_MC cost];
%         scatter(i,cost)
%         hold on
%         pause(0.0001)
    end
end

alpha = [];
if type == 3
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
            u(i,j-1) = (H(2,1)/H(2,2))*(final_state-x(i,j-1))+0.05*randn(1,1);
            K_vec = [ K_vec;(H(2,1)/H(2,2))];
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
        end
        K_vec(end)
        subplot(2,3,3)
        scatter(i,K_vec(end))
        hold on
        scatter(i,K,'.')
        hold on
        pause(0.001)
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
        H_vector = [H(1,1);H(1,2);H(2,1);H(2,2)];
        for gd_iter = 1:10
            %Calculate gradient at evaluated point
            dH_sum = zeros(4,1);
            for j=1:N_steps-1
                %For ease of notation
                matrix = [x(i,j)*x(i,j)' x(i,j)*u(i,j)' u(i,j)*x(i,j)' u(i,j)*u(i,j)'];
                %Enumerate gradient
                dH_sum = dH_sum+(matrix*H_vector-net_cost(j))*matrix';
            end
            %Calculate optimal step size at the current point
            %Use minimization rule to find alpha--
            num = 0;
            den = 0;
            for j=1:N_steps-1
                matrix = [x(i,j)^2 x(i,j)*u(i,j) x(i,j)*u(i,j) u(i,j)^2];
                vector = net_cost(j);
                num = num+ (vector*matrix*dH_sum-matrix*H_vector*matrix*dH_sum);
                den = den + (matrix*dH_sum)^2;
            end
            alpha = [alpha num/den];
            %Take a gradient step
            H_vector = H_vector+alpha(end)*dH_sum;
            subplot(2,3,1)
            scatter(gd_iter,H_vector(1),'.')
            hold on
            subplot(2,3,2)
            scatter(gd_iter,H_vector(2),'.')
            hold on
            subplot(2,3,4)
            scatter(gd_iter,H_vector(3),'.')
            hold on
            subplot(2,3,5)
            scatter(gd_iter,H_vector(4),'.')
            hold on
            pause(0.001)
        end
        %Put back in matrix
        H(1,1) = H_vector(1);
        H(1,2) = H_vector(2);
        H(2,1) = H_vector(3);
        H(2,2) = H_vector(4);
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
        pause(0.0001)
    end
end

if type == 4
    %Simulation - Q learning value gradient
    cost_MC = [];
    K_vec =[];
    x = zeros(N_episodes,N_steps);
    u = zeros(N_episodes,N_steps);
    alpha = [];
    for i=1:N_episodes
        i
        %Generate an episode
        x(i,1) = init_state;
        for j=2:N_steps
            %Apply to dynamics - Observe next state
            u(i,j-1) = (H(2,1)/H(2,2))*(final_state-x(i,j-1))+0.001*randn(1,1);
            K_vec = [ K_vec;(H(2,1)/H(2,2))];
%             %-----
            subplot(2,3,3)
            scatter(j,K_vec(end))
            hold on
            pause(0.001)
%             -----
            if x(i,j-1)<=2.5
                x(i,j) = a1*x(i,j-1)^p+b1*u(i,j-1);
            else
                x(i,j) = a2*x(i,j-1)^p+b2*u(i,j-1);
            end
            %Evaluate gradient and modify Q
            H_vector = [H(1,1);H(1,2);H(2,1);H(2,2)];
            error = final_state - x(i,j-1);
            error_next = final_state - x(i,j);
            matrix = [x(i,j-1)^2 u(i,j-1)*x(i,j-1) u(i,j-1)*x(i,j-1) u(i,j-1)^2];
            vector = (Q*error^2+R*u(i,j-1)^2) + gamma*[error_next u(i,j)]*H*[error_next;u(i,j)]; %TD(0) update
            %Sample gradient of the fitting problem
            gradient = (matrix*H_vector - vector)*matrix';
            %For this gradient, find optimum step length
            num = (vector*matrix*gradient-matrix*H_vector*matrix*gradient);
            den = (matrix*gradient)^2;
            alpha = [alpha num/den];
            %Take a gradient step to update H
            H_vector = H_vector+alpha(end)*gradient;
            %Put back in a matrix
            H(1,1) = H_vector(1);
            H(1,2) = H_vector(2);
            H(2,1) = H_vector(3);
            H(2,2) = H_vector(4);
            %Plot
            subplot(2,3,1)
            scatter(j,H_vector(1),'.')
            hold on
            subplot(2,3,2)
            scatter(j,H_vector(2),'.')
            hold on
            subplot(2,3,4)
            scatter(j,H_vector(3),'.')
            hold on
            subplot(2,3,5)
            scatter(j,H_vector(4),'.')
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
        pause(0.0001)
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


% 
% [a_grid,x_grid] = meshgrid(A,X);
% surf(a_grid,x_grid,pi_table')

