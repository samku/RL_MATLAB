function [H,U] = LS_H(x,u,E,F,gamma)

    %Get size
    nx = size(x,1);
    nu = size(u,1);
    
    %Define optimization variables of H-matrix
    H11 = sdpvar(nx,nx);
    H12 = sdpvar(nx,nu);
    H21 = sdpvar(nu,nx);
    H22 = sdpvar(nu,nu);
    H = [H11 H12; H21 H22];
    
    %We build vector of the form G(H) = g for each time step
    %G is quadratic in x's and linear in H
    G = [];
    g = [];
    %Build difference for each time step
    for i=1:size(x,2)-1
        G = [G; [x(:,i)' u(:,i)']*H*[x(:,i); u(:,i)] - gamma*[x(:,i+1)' u(:,i+1)']*H*[x(:,i+1); u(:,i+1)]];
        g = [g; [x(:,i)' u(:,i)']*[E zeros(nx,nu); zeros(nu,nx) F]*[x(:,i); u(:,i)]];
    end
    %Solve the LS problem
    optimize([],norm(G-g,2),sdpsettings('verbose',0));
    
    %Extract matrices and calculate greedy U
    H = value(H);
    U = -inv(value(H22))*value(H21);
end