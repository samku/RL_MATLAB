classdef SMD
    properties
        M = 2.0;
        c = 10.
        K = 100.
        nx = 2
        nu = 1
        dx = @(obj,x,u)...
                    [x(2);...
                     (u - obj.c*x(2)-obj.K*x(1))/obj.M];
        C = [1 0; 0 1];  %We want to do position control by giving the right voltage inputs (Should lead to steady state position with 0V)
    end
    methods
        function x_next = state_next(obj,x_curr,u_curr,Ts)
            %RK Integration
            k1 = Ts*obj.dx(obj,x_curr,u_curr);
            k2 = Ts*obj.dx(obj,x_curr+k1/2,u_curr);
            k3 = Ts*obj.dx(obj,x_curr+k2/2,u_curr);
            k4 = Ts*obj.dx(obj,x_curr+k3,u_curr);
            x_next = x_curr+(k1/6)+(k2/3)+(k3/3)+(k4/6);
        end
        function y_next = output_next(obj,x_next,noise_curr)
            y_next = obj.C*x_next+noise_curr;
        end
    end
end








