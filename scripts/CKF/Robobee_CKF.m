classdef Robobee_CKF < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        CCF
        EKF
        X
        P
    end

    methods
        function obj = Robobee_CKF(ccf,ekf)
            obj.CCF = ccf;
            obj.EKF = ekf;
            obj.X = [0 0 0 0];
            obj.P = [0 0 0 0];
        end

        function X = update(obj,acc, gyro, mag, tof, U)
            obj.CCF.update(acc, gyro, mag);
            Z_ = [obj.CCF.X; tof];

            X = obj.EKF.update(Z_, U);
            obj.CCF.X = X(1:3);
            obj.X = X;
            obj.P = obj.EKF.getConfidence();
        end
    end
end