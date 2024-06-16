classdef LTI < handle
    properties
        X;
        sys_roll;
        sys_pitch;
        sys_yaw;
        dt;
    end
    methods
        function obj = LTI(sys_roll, sys_pitch, sys_yaw, dt)
            obj.sys_roll = sys_roll;
            obj.sys_pitch = sys_pitch;
            obj.sys_yaw = sys_yaw;
            obj.X = zeros(size(sys_roll.A, 2), 3);            
            obj.dt = dt;
        end
        function Y = filter(obj, U)

%            Y = obj.sys.C*obj.X + obj.sys.D*U;
%            xdot = obj.sys.A * obj.X + obj.sys.B*U;
%            obj.X = obj.X + obj.dt*xdot;
            Y = zeros(3, 1);
            Y(1) = obj.sys_roll.C*obj.X(:, 1) + obj.sys_roll.D*U(1);
            Y(2) = obj.sys_pitch.C*obj.X(:, 2) + obj.sys_pitch.D*U(2);
            Y(3) = obj.sys_yaw.C*obj.X(:, 3) + obj.sys_yaw.D*U(3);

            xdot = obj.sys_roll.A * obj.X(:, 1) + obj.sys_roll.B*U(1);
            obj.X(:, 1) = obj.X(:, 1) + obj.dt*(xdot);

            xdot = obj.sys_pitch.A * obj.X(:, 2) + obj.sys_pitch.B*U(2);
            obj.X(:, 2) = obj.X(:, 2) + obj.dt*(xdot);

            xdot = obj.sys_yaw.A * obj.X(:, 3) + obj.sys_yaw.B*U(3);
            obj.X(:, 3) = obj.X(:, 3) + obj.dt*(xdot);
            
%            for i =1:3
%                Y(i) = obj.sys.C*obj.X(:, i) + obj.sys.D*U(i);

%                xdot = obj.sys.A * obj.X(:, i) + obj.sys.B*U(i);
%                obj.X(:, i) = obj.X(:, i) + obj.dt*(xdot);            
%            end
        end
    end
end