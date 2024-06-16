
alpha = 0.6103; % ranges from 0 to 1 weights accel vs gyro
KP = 5.2022; % Proportional gain for PI controller in NCF
KI = 1; % Integral gain for PI controller in NCF

initialGuess = [alpha, KP, KI]; % alpha KP KI

config = load('config.mat');
data = load('data/sensor_hover.mat');


options = optimset('PlotFcns',@optimplotfval, 'MaxFunEvals', 5000, 'MaxIter', 5000);
optimalParams = fminsearch(@(p) calculateCost(p, data, config), initialGuess, options);
%fprintf('Optimal alpha: %f, Kp: %f, Ki: %f\n', optimalParams(1), optimalParams(2), optimalParams(3));
optimalParams

function cost = calculateCost(params, data, config)
    filter = Robobee_CCF(params, config.sampling_time);
    traj = getEstimatedTrajectory(filter, data, config);
    cost = sum((data.Thetas(:, 1) - traj(:, 1)).^2, "all");
end

function traj = getEstimatedTrajectory(filter, data, config)

    traj = [];
    for t = 1:size(data.time)
        filter.update(data.Accelerometer(t, :), data.Gyroscope(t, :), data.Magnetometer(t, :));
    
        traj = [traj; filter.X.'];
    end

end
