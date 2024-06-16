classdef RobobeeEKF < handle
    %ROBOBEEEKF Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        H
        x
        P
        Q
        R
        dt
        config
        X
        Kp
        error
    end
    
    methods
        function obj = RobobeeEKF(params, config)
            %ROBOBEEEKF Construct an instance of this class
            %   Detailed explanation goes here
              obj.H = [ 1 0 0 0 0 0 0 0 0 0;
                    0 1 0 0 0 0 0 0 0 0;
                    0 0 1 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 1 0 0 0];
              obj.Q = diag(params(1:10));
%              obj.R = diag(params(11:14)); 
              obj.R = diag([0.07 0.07 0.07 0.002]);
              obj.x = [0 0 0 0 0 0 0 0 0 0]; 

                % state covariance
              obj.P = pi/2*eye(10);

              obj.dt = 4e-3;
              obj.config = config;
              obj.config.bw = 2e-4;
              obj.config.rw = 9e-3;

              obj.X = obj.H*obj.x.';
              obj.error = zeros(4, 1);

        end
        

        function x = update(obj, z, u, dt)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            if exist('dt','var')
                obj.dt = dt;
            end
            A = Ajacob(obj.x, u, obj.config, obj.dt);
            xp = fx(obj.x, u, obj.config, obj.dt);
            Pp = A*obj.P*A' + obj.Q;

            K = Pp*obj.H'*pinv(obj.H*Pp*obj.H' + obj.R);
            
            obj.x = xp + K*(z - obj.H*xp);
            obj.P = Pp - K*obj.H*Pp;
            x = [obj.x(1); obj.x(2); obj.x(3); obj.x(7)];
        end

        function P = getConfidence(obj)
            P = [obj.P(1, 1) obj.P(2, 2) obj.P(3, 3) obj.P(7, 7)];
        end
    end
end


function xp = fx(xhat, U, config, dt)
%
%
phi   = xhat(1);
theta = xhat(2);
psi = xhat(3);

p = xhat(4);
q = xhat(5);
r = xhat(6);

z = xhat(7);

vx = xhat(8);
vy = xhat(9);
vz = xhat(10);

F = U(4);
Tx = U(1);
Ty = U(2);
Tz = U(3);

xdot = zeros(10, 1);

v = vx*cos(psi)*cos(theta) + vy*cos(theta)*sin(psi)-vz*sin(theta);
w = p*(cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)) + q*(sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))+r*cos(theta)*sin(phi);

fd = -config.bw*(config.rw*w+v);
td = -config.rw*fd;

F_total_world = [cos(psi)*cos(theta)*fd + (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))*F;
             sin(psi)*cos(theta)*fd + (sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))*F;
             -sin(theta)*fd + cos(theta)*cos(phi)*F - config.m*config.g];


tau_total_world = [cos(psi)*cos(theta)*Tx + (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*(Ty + td) + (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))*Tz;
             sin(psi)*cos(theta)*Tx + (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(Ty + td) + (sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))*Tz;
             -sin(theta)*Tx + (cos(theta)*sin(phi))*(Ty + td) + cos(theta)*cos(phi)*Tz];

xdot(1) = p;
xdot(2) = q;
xdot(3) = r;
xdot(4) = (1/config.Ixx)*tau_total_world(1);
xdot(5) = (1/config.Iyy)*tau_total_world(2);
xdot(6) = (1/config.Izz)*tau_total_world(3);        
xdot(7) = vz;

xdot(8) = F_total_world(1)/config.m;
xdot(9) = F_total_world(2)/config.m;
xdot(10) = F_total_world(3)/config.m;

xp = xhat + xdot*dt;


end
%------------------------------
function A = Ajacob(xhat, U, config, dt)
%
% xhat = (phi theta psi wx wy wz z vx vy vz P_)
A = zeros(10, 10);

F = U(4);
Tx = U(1);
Ty = U(2);
Tz = U(3);

phi   = xhat(1);
theta = xhat(2);
psi = xhat(3);

p = xhat(4);
q = xhat(5);
r = xhat(6);

z = xhat(7);

vx = xhat(8);
vy = xhat(9);
vz = xhat(10);

A = zeros(10, 10);


v = vx*cos(psi)*cos(theta) + vy*cos(theta)*sin(psi)-vz*sin(theta);
w = p*(cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)) + q*(sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))+r*cos(theta)*sin(phi);

fd = -config.bw*(config.rw*w+v);
td = -config.rw*fd;

dtd_dfd = -config.rw;

dfd_dw = -config.bw*config.rw;
dfd_dv = -config.bw;

dv_dvx = cos(psi)*cos(theta);
dv_dvy = cos(theta)*sin(psi);
dv_dvz = -sin(theta);
dv_dtheta = -vx*cos(psi)*sin(theta) - vy*sin(theta)*sin(psi) - vz * cos(theta);
dv_dpsi = -vx*sin(psi)*cos(theta) + vy*cos(theta)*cos(psi)       ;


dw_dp = (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi));
dw_dq = (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi));
dw_dr =  cos(theta)*sin(phi);
dw_dtheta = p*(cos(psi)*cos(theta)*sin(phi)) + q*(sin(psi)*cos(theta)*sin(phi))-r*sin(theta)*sin(phi);
dw_dpsi = p*(-sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi)) + q*(cos(psi)*sin(theta)*sin(phi) + sin(psi)*cos(phi))    ;




% roll pitch yaw
A(1, 4) = 1;
A(2, 5) = 1;
A(3, 6) = 1;

% wx
A(4, 1) = (1/config.Ixx)*((Ty + td)*(cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))+ Tz*(cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)));
A(4, 2) = (1/config.Ixx)*(-Tx*(cos(psi)*sin(theta)) + (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) + (Ty + td)*(cos(psi)*cos(theta)*sin(phi)) + (cos(psi)*cos(theta)*cos(phi))*Tz);
A(4, 3) = (1/config.Ixx)*(-sin(psi)*cos(theta)*Tx + (-sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(Ty + td) + (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dpsi + dfd_dv*dv_dpsi)) + (-sin(psi)*sin(theta)*cos(phi) + cos(psi)*sin(phi))*Tz);
A(4, 4) = (1/config.Ixx)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dp);
A(4, 5) = (1/config.Ixx)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dq);
A(4, 6) = (1/config.Ixx)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dr);

A(4, 8) = (1/config.Ixx)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvx);
A(4, 9) = (1/config.Ixx)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvy);
A(4, 10) = (1/config.Ixx)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvz);

% wy
A(5, 1) = (1/config.Iyy)*((Ty + td)*(-sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)) + Tz*(-sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi)));
A(5, 2) = (1/config.Iyy)*(sin(psi)*-sin(theta)*Tx + (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) + (Ty + td)*(sin(psi)*cos(theta)*sin(phi)) + (sin(psi)*cos(theta)*cos(phi))*Tz);
A(5, 3) = (1/config.Iyy)*(cos(psi)*cos(theta)*Tx + (cos(psi)*sin(theta)*sin(phi) + sin(psi)*cos(phi))*(Ty + td) + (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dpsi + dfd_dv*dv_dpsi)) + (-cos(psi)*sin(theta)*cos(phi) - sin(psi)*sin(phi))*Tz);
A(5, 4) = (1/config.Iyy)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dp);
A(5, 5) = (1/config.Iyy)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dq);
A(5, 6) = (1/config.Iyy)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dr);

A(5, 8) = (1/config.Iyy)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvx);
A(5, 9) = (1/config.Iyy)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvy);
A(5, 10) = (1/config.Iyy)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvz);

%wz 
A(6, 1) = (1/config.Izz)*((cos(theta)*cos(phi))*(Ty + td) - cos(theta)*sin(phi)*Tz);
A(6, 2) = (1/config.Izz)*(-cos(theta)*Tx + (sin(theta)*sin(phi))*(Ty + td) + (cos(theta)*sin(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) - sin(theta)*cos(phi)*Tz);
A(6, 3) = (1/config.Izz)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dw*dw_dp);
A(6, 4) = (1/config.Izz)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dw*dw_dq);
A(6, 5) = (1/config.Izz)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dw*dw_dr);

A(6, 8) = (1/config.Izz)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dv*dv_dvx);
A(6, 9) = (1/config.Izz)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dv*dv_dvy);
A(6, 10) = (1/config.Izz)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dv*dv_dvz);

% z
A(7, 10) = 1;

% vx
A(8, 1) = (1/config.m)*((cos(psi)*sin(theta)*-sin(phi) + sin(psi)*cos(phi))*F);
A(8, 2) = (1/config.m)*(cos(psi)*-sin(theta)*fd + cos(psi)*cos(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta) + (cos(psi)*cos(theta)*cos(phi))*F);
A(8, 3) = (1/config.m)*(-sin(psi)*cos(theta)*fd  + cos(psi)*cos(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi)+ (-sin(psi)*sin(theta)*cos(phi) + cos(psi)*sin(phi))*F);
A(8, 4) = (1/config.m)*cos(psi)*cos(theta)*dfd_dw*dw_dp;
A(8, 5) = (1/config.m)*cos(psi)*cos(theta)*dfd_dw*dw_dq;
A(8, 6) = (1/config.m)*cos(psi)*cos(theta)*dfd_dw*dw_dr;

A(8, 8) = (1/config.m)*cos(psi)*cos(theta)*dfd_dv*dv_dvx;
A(8, 9) = (1/config.m)*cos(psi)*cos(theta)*dfd_dv*dv_dvy;
A(8, 10) = (1/config.m)*cos(psi)*cos(theta)*dfd_dv*dv_dvz        ;

%vy 
A(9, 1) = (1/config.m)*((cos(psi)*sin(theta)*-sin(phi)  + sin(psi)*cos(phi))*F);
A(9, 2) = (1/config.m)*(sin(psi)*-sin(theta)*fd + sin(psi)*cos(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta) + (sin(psi)*cos(theta)*cos(phi) - cos(psi)*sin(phi))*F);
A(9, 3) = (1/config.m)*(cos(psi)*cos(theta)*fd + sin(psi)*cos(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi)+ (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))*F);
A(9, 4) = (1/config.m)*sin(psi)*cos(theta)*dfd_dw*dw_dp;
A(9, 5) = (1/config.m)*sin(psi)*cos(theta)*dfd_dw*dw_dq;
A(9, 6) = (1/config.m)*sin(psi)*cos(theta)*dfd_dw*dw_dr;
A(9, 8) = (1/config.m)*sin(psi)*cos(theta)*dfd_dv*dv_dvx;
A(9, 9) = (1/config.m)*sin(psi)*cos(theta)*dfd_dv*dv_dvy;
A(9, 10) = (1/config.m)*sin(psi)*cos(theta)*dfd_dv*dv_dvz ;       

%vz 
A(10, 1) = (1/config.m) * (cos(theta)*-sin(phi)*F);
A(10, 2) = (1/config.m) * (-cos(theta)*fd  + -sin(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta) - sin(theta)*cos(phi)*F);
A(10, 3) = (1/config.m) * (-sin(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi));
A(10, 4) = (1/config.m)*-sin(theta)*dfd_dw*dw_dp;
A(10, 5) = (1/config.m)*-sin(theta)*dfd_dw*dw_dq;
A(10, 6) = (1/config.m)*-sin(theta)*dfd_dw*dw_dr;

A(10, 8) = (1/config.m)*-sin(theta)*dfd_dv*dv_dvx;
A(10, 9) = (1/config.m)*-sin(theta)*dfd_dv*dv_dvy;
A(10, 10) = (1/config.m)*-sin(theta)*dfd_dv*dv_dvz;    

A = eye(10) + A*dt;
end

