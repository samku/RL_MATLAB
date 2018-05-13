clc
clear all

%Dynamics------------------
Ts = 0.0001;
A1 = 0.9*eye(2);
B1 = Ts*eye(2);
A2 = 0.9*eye(2);
B2 = Ts*eye(2);
x1(:,1) = [35;20];
x2(:,1) = [20;20];
r = [10;5]; %Formation vector

%Problem type
type = 1; %1-distributed, 2- central

if type == 1
    %Problems--------------------
    N = 20;
    Q = 0.0001*eye(2);
    R = 0.0001*eye(2);
    Q12 = 10000*eye(2);
    %Variables on Veh 1
    x1_arr = sdpvar(2,N);
    u1_arr = sdpvar(2,N);
    x12_arr = sdpvar(2,N);
    lambda12_arr = sdpvar(2,N);
    lambda21_arr = sdpvar(2,N);
    %Problem on Veh 1
    x1_0 = sdpvar(2,1);
    f1 = 0;
    for i=1:N
        if i==1
            x1_arr(:,i) = x1_0;
        else
            x1_arr(:,i) = A1*x1_arr(:,i-1)+B1*u1_arr(:,i-1);
        end
        f1 = f1+(x1_arr(:,i)'*Q*x1_arr(:,i) + ...
                         u1_arr(:,i)'*R*u1_arr(:,i) + ...
                         (x1_arr(:,i)-x12_arr(:,i)-r)'*(Q12/2)*(x1_arr(:,i)-x12_arr(:,i)-r) + ...
                         lambda12_arr(:,i)'*x12_arr(:,i) - ...
                         lambda21_arr(:,i)'*x1_arr(:,i));
    end
    K1 = optimizer([],f1,[],{x1_0,lambda12_arr,lambda21_arr},{u1_arr,x1_arr,x12_arr});

    %Variables on Veh 2
    x2_arr = sdpvar(2,N);
    u2_arr = sdpvar(2,N);
    x21_arr = sdpvar(2,N);
    lambda12_arr = sdpvar(2,N);
    lambda21_arr = sdpvar(2,N);
    %Problem on Veh 1
    x2_0 = sdpvar(2,1);
    f2 = 0;
    for i=1:N
        if i==1
            x2_arr(:,i) = x2_0;
        else
            x2_arr(:,i) = A2*x2_arr(:,i-1)+B2*u2_arr(:,i-1);
        end
        f2 = f2+(x2_arr(:,i)'*Q*x2_arr(:,i) + ...
                         u2_arr(:,i)'*R*u2_arr(:,i) + ...
                         (x21_arr(:,i)-x2_arr(:,i)-r)'*(Q12/2)*(x21_arr(:,i)-x2_arr(:,i)-r) + ...
                         lambda21_arr(:,i)'*x21_arr(:,i) - ...
                         lambda12_arr(:,i)'*x2_arr(:,i));
    end
    K2 = optimizer([],f2,[],{x2_0,lambda12_arr,lambda21_arr},{u2_arr,x2_arr,x21_arr});

    %Simulation----------------------------------
    N_sim = 200;
    dd_iters = 5;
    alpha = 100;
    lambda12 = zeros(2,N);
    lambda21 = zeros(2,N);
    figure
    for i=2:N_sim
        i
        for j=1:dd_iters
            %Solve local problems
            a1 = K1{{x1(:,i-1),lambda12,lambda21}}; %Vehicle 1
            a2 = K2{{x2(:,i-1),lambda12,lambda21}}; %Vehicle 2
            %Use communication
            x1_sol = a1{2}; %Sent to veh 2
            x12_sol = a1{3}; %Local to veh 1
            x2_sol = a2{2}; %Sent to veh 1
            x21_sol = a2{3}; %Local to veh 2
            %Update lambdas
            lambda12 = lambda12 +alpha*(x12_sol-x2_sol);
            lambda21 = lambda21 + alpha*(x21_sol-x1_sol);
            %Check consensus
    %         for k=1:N
    %             scatter(x1_sol(1,k),x1_sol(2,k),'.','red');
    %             hold on
    %             scatter(x21_sol(1,k),x21_sol(2,k),'.','black');
    %             pause(0.001)
    %         end
        end
        %Apply to dynamics
        u1(:,i-1) = a1{1}(:,1);
        x1(:,i) = A1*x1(:,i-1)+B1*u1(:,i-1);
        u2(:,i-1) = a2{1}(:,1);
        x2(:,i) = A2*x2(:,i-1)+B2*u2(:,i-1);
        %Plot trajectories
        for k=1:i
            scatter(x1(1,k),x1(2,k),10,'o','red');
            hold on
            scatter(x2(1,k),x2(2,k),10,'o','black');
            hold on
        end
        plot(x1_sol(1,:),x1_sol(2,:),'red','LineWidth',0.01)
        hold on
        plot(x2_sol(1,:),x2_sol(2,:),'black','LineWidth',0.01)
        pause(0.001)
        hold off
%         scatter(i,x1(1,i)-x2(1,i),'.','red');
%         hold on
%         scatter(i,x1(2,i)-x2(2,i),'.','black');
%         hold on
%         pause(0.001)
    end
    
    
else
    %%%%%%%%%%%%%%%%%%%
    %Centralized problem
    N = 2;
    Q = 1*eye(2);
    R = 0.0001*eye(2);
    Q12 = 1000*eye(2);
    %Variables
    x1_arr = sdpvar(2,N);
    u1_arr = sdpvar(2,N);
    x2_arr = sdpvar(2,N);
    u2_arr = sdpvar(2,N);
    %Problem on Veh 1
    x_0 = sdpvar(4,1);
    f = 0;
    for i=1:N
        if i==1
            x1_arr(:,i) = x_0(1:2,1);
            x2_arr(:,i) = x_0(3:4,1);
        else
            x1_arr(:,i) = A1*x1_arr(:,i-1)+B1*u1_arr(:,i-1);
            x2_arr(:,i) = A2*x2_arr(:,i-1)+B2*u2_arr(:,i-1);
        end
        f = f+(x1_arr(:,i)'*Q*x1_arr(:,i) + x2_arr(:,i)'*Q*x2_arr(:,i) + ...
                   u1_arr(:,i)'*R*u1_arr(:,i) + u2_arr(:,i)'*R*u2_arr(:,i) +...
                   (x1_arr(:,i)-x2_arr(:,i)-r)'*(Q12)*(x1_arr(:,i)-x2_arr(:,i)-r));
    end
    K_central = optimizer([],f,[],{x_0},{u1_arr,u2_arr,x1_arr,x2_arr});

    %Simulation----------------------------------
    N_sim = 200;
    figure
    for i=2:N_sim
        i
        %Solve local problems
        a = K_central{{[x1(:,i-1);x2(:,i-1)]}};
        %Apply to dynamics
        u1(:,i-1) = a{1}(:,1);
        x1(:,i) = A1*x1(:,i-1)+B1*u1(:,i-1);
        u2(:,i-1) = a{2}(:,1);
        x2(:,i) = A2*x2(:,i-1)+B2*u2(:,i-1);
        %Plot trajectories
            scatter(x1(1,i),x1(2,i),'.','red');
            hold on
            scatter(x2(1,i),x2(2,i),'.','black');
            hold on
%         scatter(i,x1(1,i)-x2(1,i),'.','red');
%         hold on
%         scatter(i,x1(2,i)-x2(2,i),'.','black');
%         hold on
        pause(0.001)
    end
    
end