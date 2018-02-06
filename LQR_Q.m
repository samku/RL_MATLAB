clc
clear all

%Load system
sys = SMD();
nx = sys.nx;
nu = sys.nu;
x0 = zeros(sys.nx,1);
x0(1) = 10;
Ts = 0.0001;

%Number of episodes
epsds = 10;

%Duration per episode
N = 1000;

%Initial feedback gain U
U = -0.01*ones(1,nx);

%Procedure
%For LQR, Q(x_t,u_t) = r(x_t,u_t)  + gamma*Q(x_(t+1),u_(t+1))
%Following usual steps, first do policy iteration to find how good U is
%Then do U = eps-greedy(Q)
%For LQR, Q(x_t,u_t) = [x_t; u_t]'*[H11 H12; H21 H22]*[x_t; u_t] - H is an
%unknown matrix, dependedent on system
%FOr LQR, r(x_t,u_t) = x_t'*E*x_t + u_t'*F*u_t - Known value
%So we can assemble all r's and in an LS sense, find H.

%LQR costs
E= eye(nx);
F = eye(nu);

%Discount factor
gamma = 1.0;

%Perform experiments and improvement
figure
for eps = 1:epsds
    %Store vectors
    x_vec{eps} = x0;
    u_vec{eps} = U*x0;
    cost_LQR{eps} = 0;
    for i=2:N
        %Propogate state
        x_vec{eps} = [x_vec{eps}, state_next(sys,x_vec{eps}(:,end),u_vec{eps}(:,end),Ts)];
        u_vec{eps} = [u_vec{eps}, U*x_vec{eps}(:,end)];
        %Calculate net cost
        cost_LQR{eps} = cost_LQR{eps}+[x_vec{eps}(:,i)' u_vec{eps}(:,i)']*[E zeros(nx,nu); zeros(nu,nx) F]*[x_vec{eps}(:,i); u_vec{eps}(:,i)];
    end
    %Calculate new U based on data
    U = LS_H(x_vec{eps},u_vec{eps},E,F,gamma);
    %Plotting
    subplot(2,1,1)
    plot(x_vec{eps}(1,:))
    grid on
    hold on
    subplot(2,1,2)
    plot(u_vec{eps}(1,:))
    grid on
    pause(0.001)
    hold on
end


