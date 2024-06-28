classdef Robobee_CCF < handle
    properties 
        X = [0 0 0]; 

        LTI_HPS;
        
        LTI_LPS;

        dt;
    end
    methods
        function obj = Robobee_CCF(params, dt)
            alphas = params(1:3);
            KP = params(4:6);
            KI = params(7:9);

            obj.X = [0; 0; 0];
            obj.dt = dt;

            s = tf('s');
            
            HPF_roll_ss = ss((1/s)*(alphas(1)*s^2)/(s^2 + alphas(1)*KP(1)*s + alphas(1)*KI(1)));
            HPF_pitch_ss = ss((1/s)*(alphas(2)*s^2)/(s^2 + alphas(2)*KP(2)*s + alphas(2)*KI(2)));
            HPF_yaw_ss = ss((1/s)*(alphas(3)*s^2)/(s^2 + alphas(3)*KP(3)*s + alphas(3)*KI(3)));
            obj.LTI_HPS = LTI(HPF_roll_ss, HPF_pitch_ss, HPF_yaw_ss, dt);

            LPF_roll_ss = ss(((1 - alphas(1))*s^2 + alphas(1)*KP(1)*s+alphas(1)*KI(1))/(s^2 + alphas(1)*KP(1)*s + alphas(1)*KI(1)));
            LPF_pitch_ss = ss(((1 - alphas(2))*s^2 + alphas(2)*KP(2)*s+alphas(2)*KI(2))/(s^2 + alphas(2)*KP(2)*s + alphas(2)*KI(2)));
            LPF_yaw_ss = ss(((1 - alphas(3))*s^2 + alphas(3)*KP(3)*s+alphas(3)*KI(3))/(s^2 + alphas(3)*KP(3)*s + alphas(3)*KI(3)));
            obj.LTI_LPS = LTI(LPF_roll_ss, LPF_pitch_ss, LPF_yaw_ss, dt);

        end

        function update(obj, acc, gyro, mag, tof, u)
            X_g_dot_ = obj.gyro_estimate(gyro);
            g = 9.81;            
            if norm(acc) > 4*g
                norm(acc);
                obj.X = X_g_dot_*obj.dt;
            else
                X_g = obj.LTI_HPS.filter(X_g_dot_);
                X_a_ = obj.accel_mag_estimate(acc, mag);
                X_a = obj.LTI_LPS.filter(X_a_);
           
                obj.X = X_g + X_a;
            end


        end
        
        function X_a = accel_mag_estimate(obj, acc, mag)        
            phi = obj.X(1);
            theta = obj.X(2);
            X_a = zeros(3, 1);
            X_a(1) = atan2(acc(2), acc(3));
            X_a(2) = atan2(-acc(1), acc(2)*sin(phi) + acc(3)*cos(phi));
            X_a(3) = atan2(mag(3)*sin(phi) - mag(2)*cos(phi), mag(1)*cos(theta) + mag(2)*sin(theta)*sin(phi) + mag(3)*sin(theta)*cos(phi)); 
        end

        function X_g_dot = gyro_estimate(obj, gyro)
            X_g_dot = zeros(3, 1);
            gyro = (pi/180)*gyro;

            phi = obj.X(1);
            theta = obj.X(2);
        
            X_g_dot(1) = gyro(1) + gyro(2)*sin(phi)*tan(theta) + gyro(3)*cos(phi)*tan(theta);
            X_g_dot(2) = gyro(2)*cos(phi) - gyro(3)*sin(theta);
            X_g_dot(3) = gyro(1) + gyro(2)*sin(phi)*sec(theta) + gyro(3)*cos(phi)*sec(theta);
        end
    end
end