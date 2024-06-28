close all; clear all;

addpath('CCF');      
addpath('EKF');
addpath('CKF');

config = load('data/config.mat');

folder = 'data/simulation/sensors';
DirList = dir(fullfile(folder, '*.mat'));
RMSE = [];

alpha = [0.5371 0.7081 0.0500];
KP = [0.0001 1.4592 100];
KI = [0.0001 32.4265 100];



Q = ones(10, 1);
CCFparams = [alpha KP KI];

max_ = [];
for i = 1:size(DirList, 1)
    data = load(fullfile(folder, DirList(i).name));
    ccf = Robobee_CCF(CCFparams, 4e-3);
    ekf = RobobeeEKF(Q, config);
    ckf = Robobee_CKF(ccf, ekf);
    [traj] = getEstimatedTrajectory(ckf, data);
    [trueTraj, t] = getTrueTraj(data, 1e-4, 4e-3);
    max_ = [max_; max(abs(trueTraj()*180/pi))];
    RMSE = [RMSE; getRMSE(trueTraj, traj)];
end

max(max_)
mean(RMSE)*180/pi
median(RMSE)*180/pi
std(RMSE)*180/pi


function RMSE = getRMSE(true, traj)
    error = wrapToPi(wrapToPi(true) - wrapToPi(traj));
    RMSE = sqrt(mean(error.^2));
end

function [traj] = getEstimatedTrajectory(filter, data)

    traj = [];
    for t = 1:40:size(data.time)
        filter.update(data.Accelerometer(t, :), data.Gyroscope(t, :), data.Magnetometer(t, :), data.TOF(t), data.U(t, :));
        traj = [traj; filter.X.'];
    end

end


function [trueTraj, t] = getTrueTraj(data, original_dt, new_dt)
    s = new_dt/original_dt;
    trueTraj = [data.Thetas(1:s:end, :) data.trueZ(1:s:end, :) - 0.015];
    t = data.time(1:s:end);
end
