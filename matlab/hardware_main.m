close all; clear all;

addpath('CCF');      % include quaternion library
addpath('EKF');
addpath('CKF');
addpath('utils');

config = load('data/config.mat');

Q =  ones(1, 10);

flight_experiment = "leaf_hopping";
pwm = 30;

file = 'data/hardware/' + flight_experiment +'/pwm_' + string(pwm) + '.csv';

flight_data = csvread(file,1, 0);


ekf = RobobeeEKF(Q, config);
traj = getEstimatedTrajectory(ekf, flight_data);
[trueTraj, t] = getTrueTraj(flight_data);

getRMSE(trueTraj, traj)


function RMSE = getRMSE(true, traj)  
    error = wrapToPi(wrapToPi(true(:, 1:3)) - wrapToPi(traj(:, 1:3)));
    error = [error true(:, 4) - traj(:, 4)];
    RMSE = sqrt(mean(error.^2));
    RMSE(1:3) =  RMSE(1:3)*180/pi;
end

function traj = getEstimatedTrajectory(filter, data)
    traj = [];
    for t = 1:size(data, 1)
        imu = data(t, 2:4);
        imu(3) = imu(3);
        meas = [imu*(pi/180) data(t, 5)/1000].';
        
        u = data(t, 18:21);
        if t > 1
            x = filter.update(meas, u, data(t, 1) - data(t-1, 1));
        else
            x = filter.update(meas, u);
        end

        traj = [traj; x.'];
    end

end

function [trueTraj, t] = getTrueTraj(data)
    trueTraj = [data(:, 9:11) data(:, 8)];
    peak2peak(trueTraj(:, 1))
    t = data(:, 1);
end


