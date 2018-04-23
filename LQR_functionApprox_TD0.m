clc
clear all

%Load system
sys = SMD();
nx = sys.nx;
nu = sys.nu;
x0 = zeros(sys.nx,1);
x0(1) = 10;
Ts = 0.01;

%Number of episodes
epsds = 10000;

%Duration per episode
N = 800;

%Initial feedback gain U
U = -0.1*ones(1,nx);

%LQR costs
E= eye(nx)*10;
F = eye(nu)*0.01;

%Procedure
%For LQR, Q(x_t,u_t) = r(x_t,u_t)  + gamma*Q(x_(t+1),u_(t+1))
%Following usual steps, first do policy iteration to find how good U is
%Then do U = eps-greedy(Q)
%For LQR, Q(x_t,u_t) = [x_t; u_t]'*[H11 H12; H21 H22]*[x_t; u_t] - H is an
%unknown matrix, dependedent on system
%FOr LQR, r(x_t,u_t) = x_t'*E*x_t + u_t'*F*u_t - Known value
%So we can assemble all r's and in an LS sense, find H.

%Compute final optimal result for the SMD system--------------------------
M = sys.M;
c = sys.c;
K = sys.K;
A = [0 1; -K/M -c/M];
B = [0;1/M];
Ad = eye(nx)+A*Ts;
Bd = B*Ts;
[K,S,e] = lqr(A,B,E,F,0);
x = [10;0];
for i=2:N
    x = [x, (Ad-Bd*K)*x(:,end)];
end
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,2)
plot(x(1,:),':','LineWidth',2)
title('Position')
grid on
hold on
subplot(2,2,4)
plot(x(2,:),':','LineWidth',2)
title('Velocity')
grid on
hold on
subplot(2,2,1)
plot(-K*x,':','LineWidth',2)
title('Force In')
grid on
hold on
subplot(2,2,3)
title('LQ cost')
xlabel('Episode')
grid on
hold on
%-------------------------------------------------------------------------------------
%RL parameters
gamma = 1.0;
W = ones(6,1);
dW = zeros(size(W));
alpha = 1e-8;
noise = 1.0;
feature = @(x,u)[x(1)^2; x(2)^2; x(1)*x(2); x(1)*u; x(2)*u; u^2];

%Perform experiments and improvement
for eps = 1:epsds
    %Calculate FB gain from W vector
    U = (-0.5/W(6))*[W(4) W(5)];
    %Store vectors
    x_vec{eps} = x0;
    u_vec{eps} = U*x0;
    cost_LQR{eps} = 0;
    for i=2:N
        %Propogate state
        x_vec{eps} = [x_vec{eps}, state_next(sys,x_vec{eps}(:,end),u_vec{eps}(:,end),Ts)];
        u_vec{eps} = [u_vec{eps}, U*x_vec{eps}(:,end)+noise*randn(nu)];
        %Calculate net cost
        cost_LQR{eps} = cost_LQR{eps}+ [x_vec{eps}(:,i)' u_vec{eps}(:,i)']*[E zeros(nx,nu); zeros(nu,nx) F]*[x_vec{eps}(:,i); u_vec{eps}(:,i)];
        %Find TD(0) for (i-1) and update gradient of H
        prev_stg_cost = [x_vec{eps}(:,i)' u_vec{eps}(:,i)']*[E zeros(nx,nu); zeros(nu,nx) F]*[x_vec{eps}(:,i); u_vec{eps}(:,i)];
        Q_pi = prev_stg_cost + gamma*W'*feature(x_vec{eps}(:,i), u_vec{eps}(:,i));
        Q_W = W'*feature(x_vec{eps}(:,i-1), u_vec{eps}(:,i-1));
        TD = Q_pi-Q_W;
        dW  =  2*TD*feature(x_vec{eps}(:,i-1), u_vec{eps}(:,i-1));
        W = W - alpha*dW;
    end
    %Plotting
    subplot(2,2,2)
    plot(x_vec{eps}(1,:))
    hold on
    subplot(2,2,4)
    plot(x_vec{eps}(2,:))
    hold on
    subplot(2,2,1)
    plot(u_vec{eps}(1,:))
    hold on
    if eps>1
        subplot(2,2,3)
        line([eps-1,eps],[cost_LQR{eps-1},cost_LQR{eps}])
        hold on
    end
%     subplot(2,2,3)
%     yyaxis left
%     plot(cost)
%     hold on
%     subplot(2,2,3)
%     yyaxis right
%     plot(prob,':')
%     hold on
    pause(0.001)
end