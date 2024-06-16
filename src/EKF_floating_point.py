import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import scipy as sc

        
class RobobeeEKF:
    def __init__(self) -> None:
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        self.Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.R = np.diag([0.07, 0.07, 0.07, 0.002])
        self.x = np.zeros((10, 1)) # roll pitch yaw p q r z vx vy vz
        self.P = np.pi/2 * np.eye(10)

        self.dt = 4e-3
        self.bw = 2e-4
        self.rw = 9e-3

        self.I = np.array([1.42e-9, 1.34e-9, 4.5e-10])

        self.m = 8.6e-05

        self.dtype = 'float64'

    def updatedtype(self, dtype):
        self.H = self.H.astype(dtype)
        self.Q = self.Q.astype(dtype)
        self.R = self.R.astype(dtype)
        self.x = self.x.astype(dtype)
        self.P = self.P.astype(dtype)
        self.I = self.I.astype(dtype)


        # if dtype == 'float16' or dtype=='float8':
        #     self.I = np.array([1e-4, 1e-4, 1e-7]).astype('float16')
        self.dtype = dtype



    def update(self, z, u, dt=None) -> np.ndarray:
        if dt: self.dt = dt

        A = self.jacobian(self.x, u).astype(self.dtype) # 71 mult 15 add       
        xp = self.fx(self.x, u).astype(self.dtype) # 39 mult 16 add
        Pp = A @ self.P @ A.T + self.Q # 635 mult 10 add
        Pp = Pp.astype(self.dtype)        

        G = self.H @ Pp @ self.H.T + self.R # 4 addition
        G = G.astype(self.dtype)


        K = Pp @ self.H.T @ sc.linalg.pinv(G)  # 100 mult 160 additions
        K = K.astype(self.dtype)

        e = z - self.H @ xp # 4 addition
        e = e.astype(self.dtype)


        self.x = xp + K @ e # 16 mult 16 add
        self.P = Pp - K @ self.H @ Pp

        self.x = self.x.astype(self.dtype)
        self.P = self.P.astype(self.dtype)
        return self.H @ self.x


        
    def fx(self, xhat, u) -> np.ndarray:
        phi = xhat[0, 0]
        theta = xhat[1, 0]
        psi = xhat[2, 0]

        p = xhat[3, 0]
        q = xhat[4, 0]
        r = xhat[5, 0]

        z = xhat[6, 0]

        vx = xhat[7, 0]
        vy = xhat[8, 0]
        vz = xhat[9, 0]

        Ixx = self.I[0]
        Iyy = self.I[1]
        Izz = self.I[2]

        F = u[3]
        Tx = u[0]
        Ty = u[1]
        Tz = u[2]

        xdot = np.zeros((10, 1))


        g = 9.8


        v = vx*np.cos(psi)*np.cos(theta) + vy*np.cos(theta)*np.sin(psi)-vz*np.sin(theta)
        w = p*(np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi)) + q*(np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))+r*np.cos(theta)*np.sin(phi)

        fd = -self.bw*(self.rw*w+v)
        td = -self.rw*fd

        F_total_world = [np.cos(psi)*np.cos(theta)*fd + (np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi))*F,
                         np.sin(psi)*np.cos(theta)*fd + (np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi))*F,
                         -np.sin(theta)*fd + np.cos(theta)*np.cos(phi)*F - self.m*g]


        tau_total_world = [np.cos(psi)*np.cos(theta)*Tx + (np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*(Ty + td) + (np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi))*Tz,
                         np.sin(psi)*np.cos(theta)*Tx + (np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*(Ty + td) + (np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi))*Tz,
                         -np.sin(theta)*Tx + (np.cos(theta)*np.sin(phi))*(Ty + td) + np.cos(theta)*np.cos(phi)*Tz]

        xdot[0] = p
        xdot[1] = q
        xdot[2] = r
        xdot[3] = (1/Ixx)*tau_total_world[0]
        xdot[4] = (1/Iyy)*tau_total_world[1]
        xdot[5] = (1/Izz)*tau_total_world[2]        
        xdot[6] = vz

        xdot[7] = F_total_world[0]/self.m
        xdot[8] = F_total_world[1]/self.m
        xdot[9] = F_total_world[2]/self.m


        return xhat + xdot*self.dt # 1 mult 1 add
    


    def jacobian(self, xhat, u) -> np.ndarray:
        A = np.zeros((10, 10))

        phi = xhat[0, 0]
        theta = xhat[1, 0]
        psi = xhat[2, 0]

        p = xhat[3, 0]
        q = xhat[4, 0]
        r = xhat[5, 0]

        z = xhat[6, 0]

        vx = xhat[7, 0]
        vy = xhat[8, 0]
        vz = xhat[9, 0]

        Ixx = self.I[0]
        Iyy = self.I[1]
        Izz = self.I[2]


        Tx = u[0]
        Ty = u[1]
        Tz = u[2]
        F = u[3]


        v = vx*np.cos(psi)*np.cos(theta) + vy*np.cos(theta)*np.sin(psi)-vz*np.sin(theta)
        w = p*(np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi)) + q*(np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))+r*np.cos(theta)*np.sin(phi)

        fd = -self.bw*(self.rw*w+v)
        td = -self.rw*fd

        dtd_dfd = -self.rw
        
        dfd_dw = -self.bw*self.rw
        dfd_dv = -self.bw
        
        dv_dvx = np.cos(psi)*np.cos(theta)
        dv_dvy = np.cos(theta)*np.sin(psi)
        dv_dvz = -np.sin(theta)
        dv_dtheta = -vx*np.cos(psi)*np.sin(theta) - vy*np.sin(theta)*np.sin(psi) - vz * np.cos(theta)
        dv_dpsi = -vx*np.sin(psi)*np.cos(theta) + vy*np.cos(theta)*np.cos(psi)       


        dw_dp = (np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))
        dw_dq = (np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))
        dw_dr =  np.cos(theta)*np.sin(phi)
        dw_dtheta = p*(np.cos(psi)*np.cos(theta)*np.sin(phi)) + q*(np.sin(psi)*np.cos(theta)*np.sin(phi))-r*np.sin(theta)*np.sin(phi)
        dw_dpsi = p*(-np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi)) + q*(np.cos(psi)*np.sin(theta)*np.sin(phi) + np.sin(psi)*np.cos(phi))    


        A[0, 3] = 1
        A[1, 4] = 1
        A[2, 5] = 1



        A[3, 0] = (1/Ixx)*((Ty + td)*(np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)) 
                           + Tz*(np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)))
        A[3, 1] = (1/Ixx)*(-Tx*(np.cos(psi)*np.sin(theta)) 
                           + (np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) 
                           + (Ty + td)*(np.cos(psi)*np.cos(theta)*np.sin(phi))
                           + (np.cos(psi)*np.cos(theta)*np.cos(phi))*Tz)
        A[3, 2] = (1/Ixx)*(-np.sin(psi)*np.cos(theta)*Tx
                           + (-np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*(Ty + td)
                           + (np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*(dtd_dfd*(dfd_dw*dw_dpsi + dfd_dv*dv_dpsi))
                           + (-np.sin(psi)*np.sin(theta)*np.cos(phi) + np.cos(psi)*np.sin(phi))*Tz)
        A[3, 3] = (1/Ixx)*((np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*dtd_dfd*dfd_dw*dw_dp)
        A[3, 4] = (1/Ixx)*((np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*dtd_dfd*dfd_dw*dw_dq)
        A[3, 5] = (1/Ixx)*((np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*dtd_dfd*dfd_dw*dw_dr)

        A[3, 7] = (1/Ixx)*((np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*dtd_dfd*dfd_dv*dv_dvx)
        A[3, 8] = (1/Ixx)*((np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*dtd_dfd*dfd_dv*dv_dvy)
        A[3, 9] = (1/Ixx)*((np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*dtd_dfd*dfd_dv*dv_dvz)



        A[4, 0] = (1/Iyy)*((Ty + td)*(-np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)) 
                           + Tz*(-np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi)))
        A[4, 1] = (1/Iyy)*(np.sin(psi)*-np.sin(theta)*Tx 
                           + (np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) 
                           + (Ty + td)*(np.sin(psi)*np.cos(theta)*np.sin(phi))
                           + (np.sin(psi)*np.cos(theta)*np.cos(phi))*Tz)
        A[4, 2] = (1/Iyy)*(np.cos(psi)*np.cos(theta)*Tx
                           + (np.cos(psi)*np.sin(theta)*np.sin(phi) + np.sin(psi)*np.cos(phi))*(Ty + td)
                           + (np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*(dtd_dfd*(dfd_dw*dw_dpsi + dfd_dv*dv_dpsi))
                           + (-np.cos(psi)*np.sin(theta)*np.cos(phi) - np.sin(psi)*np.sin(phi))*Tz)
        A[4, 3] = (1/Iyy)*((np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*dtd_dfd*dfd_dw*dw_dp)
        A[4, 4] = (1/Iyy)*((np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*dtd_dfd*dfd_dw*dw_dq)
        A[4, 5] = (1/Iyy)*((np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*dtd_dfd*dfd_dw*dw_dr)

        A[4, 7] = (1/Iyy)*((np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*dtd_dfd*dfd_dv*dv_dvx)
        A[4, 8] = (1/Iyy)*((np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*dtd_dfd*dfd_dv*dv_dvy)
        A[4, 9] = (1/Iyy)*((np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi))*dtd_dfd*dfd_dv*dv_dvz)


        A[5, 0] = (1/Izz)*((np.cos(theta)*np.cos(phi))*(Ty + td)
                            - np.cos(theta)*np.sin(phi)*Tz)
        A[5, 1] = (1/Izz)*(-np.cos(theta)*Tx 
                           + (np.sin(theta)*np.sin(phi))*(Ty + td)
                           + (np.cos(theta)*np.sin(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) 
                           - np.sin(theta)*np.cos(phi)*Tz)
        A[5, 3] = (1/Izz)*((np.cos(theta)*np.sin(phi))*dtd_dfd*dfd_dw*dw_dp)
        A[5, 4] = (1/Izz)*((np.cos(theta)*np.sin(phi))*dtd_dfd*dfd_dw*dw_dq)
        A[5, 5] = (1/Izz)*((np.cos(theta)*np.sin(phi))*dtd_dfd*dfd_dw*dw_dr)

        A[5, 7] = (1/Izz)*((np.cos(theta)*np.sin(phi))*dtd_dfd*dfd_dv*dv_dvx)
        A[5, 8] = (1/Izz)*((np.cos(theta)*np.sin(phi))*dtd_dfd*dfd_dv*dv_dvy)
        A[5, 9] = (1/Izz)*((np.cos(theta)*np.sin(phi))*dtd_dfd*dfd_dv*dv_dvz)

        A[6, 9] = 1


        A[7, 0] = (1/self.m)*((np.cos(psi)*np.sin(theta)*-np.sin(phi) 
                               + np.sin(psi)*np.cos(phi))*F)
        A[7, 1] = (1/self.m)*(np.cos(psi)*-np.sin(theta)*fd
                              + np.cos(psi)*np.cos(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta)
                              + (np.cos(psi)*np.cos(theta)*np.cos(phi))*F)
        A[7, 2] = (1/self.m)*(-np.sin(psi)*np.cos(theta)*fd 
                              + np.cos(psi)*np.cos(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi)
                              + (-np.sin(psi)*np.sin(theta)*np.cos(phi) + np.cos(psi)*np.sin(phi))*F)
        A[7, 3] = (1/self.m)*np.cos(psi)*np.cos(theta)*dfd_dw*dw_dp
        A[7, 4] = (1/self.m)*np.cos(psi)*np.cos(theta)*dfd_dw*dw_dq
        A[7, 5] = (1/self.m)*np.cos(psi)*np.cos(theta)*dfd_dw*dw_dr

        A[7, 7] = (1/self.m)*np.cos(psi)*np.cos(theta)*dfd_dv*dv_dvx
        A[7, 8] = (1/self.m)*np.cos(psi)*np.cos(theta)*dfd_dv*dv_dvy
        A[7, 9] = (1/self.m)*np.cos(psi)*np.cos(theta)*dfd_dv*dv_dvz        

        A[8, 0] = (1/self.m)*((np.cos(psi)*np.sin(theta)*-np.sin(phi) 
                               + np.sin(psi)*np.cos(phi))*F)
        A[8, 1] = (1/self.m)*(np.sin(psi)*-np.sin(theta)*fd
                              + np.sin(psi)*np.cos(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta)
                              + (np.sin(psi)*np.cos(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi))*F)
        A[8, 2] = (1/self.m)*(np.cos(psi)*np.cos(theta)*fd
                              + np.sin(psi)*np.cos(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi)
                              + (np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi))*F)
        A[8, 3] = (1/self.m)*np.sin(psi)*np.cos(theta)*dfd_dw*dw_dp
        A[8, 4] = (1/self.m)*np.sin(psi)*np.cos(theta)*dfd_dw*dw_dq
        A[8, 5] = (1/self.m)*np.sin(psi)*np.cos(theta)*dfd_dw*dw_dr

        A[8, 7] = (1/self.m)*np.sin(psi)*np.cos(theta)*dfd_dv*dv_dvx
        A[8, 8] = (1/self.m)*np.sin(psi)*np.cos(theta)*dfd_dv*dv_dvy
        A[8, 9] = (1/self.m)*np.sin(psi)*np.cos(theta)*dfd_dv*dv_dvz        

        A[9, 0] = (1/self.m) * (np.cos(theta)*-np.sin(phi)*F)
        A[9, 1] = (1/self.m) * (-np.cos(theta)*fd 
                                + -np.sin(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta)
                                - np.sin(theta)*np.cos(phi)*F)
        A[9, 2] = (1/self.m) * (-np.sin(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi))
        A[9, 3] = (1/self.m)*-np.sin(theta)*dfd_dw*dw_dp
        A[9, 4] = (1/self.m)*-np.sin(theta)*dfd_dw*dw_dq
        A[9, 5] = (1/self.m)*-np.sin(theta)*dfd_dw*dw_dr

        A[9, 7] = (1/self.m)*-np.sin(theta)*dfd_dv*dv_dvx
        A[9, 8] = (1/self.m)*-np.sin(theta)*dfd_dv*dv_dvy
        A[9, 9] = (1/self.m)*-np.sin(theta)*dfd_dv*dv_dvz    


        return np.eye(10) + self.dt*A # 25 mults 10 adds



def getEstimatedTrajectory(ekf, data):
    T = data.shape[0]
    estimated_trajectory = np.zeros((T, 4))
    for t in tqdm(range(T)):
        imu = data[t, 1:4].reshape(3, 1)

        meas = np.zeros((4, 1))
        meas[:3] = (np.pi/180)*imu
        meas[3] = 1e-3*data[t, 4] 

        u = data[t, 17:21]


        prevt = 0
        if t > 0:
            estimated_trajectory[t, :] = ekf.update(meas, u, data[t, 0] - data[t-1, 0]).reshape(4, )
        else:
            estimated_trajectory[t, :] = ekf.update(meas, u).reshape(4, )
        
    return estimated_trajectory


def getGroundTruth(data):
    gt = np.zeros((data.shape[0], 4))
    gt[:, :3] = data[:, 8:11]
    gt[:, 3] = data[:, 7]
    t = data[:, 0]
    return gt, t


def getRMSE(true, traj):
    error = np.zeros(true.shape)

    error[:, :3] = 180/np.pi*(true[:, :3] - traj[:, :3])
    error[:, 3] = true[:, 3] - traj[:, 3]

    return np.sqrt(np.mean(np.power(error, 2), axis=0))


if __name__ == '__main__':
    ekf = RobobeeEKF()
    ekf.updatedtype('float32')
    data = np.genfromtxt("golden.csv", delimiter=",")[1:, :]
    estimated_trajectory = getEstimatedTrajectory(ekf, data)
    true_trajectory, time = getGroundTruth(data)
    RMSE32 = getRMSE(true_trajectory, estimated_trajectory)
    print("float 32")
    print(RMSE32)

    ekf = RobobeeEKF()
    ekf.updatedtype('float64')
    data = np.genfromtxt("golden.csv", delimiter=",")[1:, :]
    estimated_trajectory = getEstimatedTrajectory(ekf, data)
    true_trajectory, time = getGroundTruth(data)
    RMSE64 = getRMSE(true_trajectory, estimated_trajectory)
    print(RMSE64)
    print("float 64")