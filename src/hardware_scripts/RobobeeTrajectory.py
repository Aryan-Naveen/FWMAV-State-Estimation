import rtde_control
import rtde_receive
import numpy as np
import time
from tqdm import tqdm
import serial
import time
import threading
import queue
import pandas as pd
import os

readData = True

global q
q = queue.Queue(0)

global buffer
buffer = ''

SENSOR_ON = False

trajectory = "hovering_trajectory"



### CHANGE HERE 
is_leaf_traj = True
pwm = 10
trial = 3

if is_leaf_traj: trajectory = "leaf_hopping_trajectory"


output_dir = f"experiment_results/{trajectory}/pwm_{pwm}/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = f"{output_dir}trial_{trial}.csv"



def serial_reader(ser, output_file, rtde_r):
    # Open the output file in append mode
    global buffer
    while readData:
        buffer += ser.read(ser.inWaiting() or 1).decode()
        if '\n' in buffer: #split data line by line and store it in var
            var, buffer = buffer.split('\n', 1)
            t = rtde_r.getTimestamp()
            q.put(f'{t} {var}') #put received line in the queue

#Setup Robot with IP adress

rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.2")


if is_leaf_traj:
    traj = np.loadtxt("leaf_hopping.csv", delimiter=",", dtype=float)
else:
    traj = np.loadtxt("hovering.csv", delimiter=",", dtype=float)

record_variables = ["timestamp", "actual_TCP_pose"]


# UR5e Speeds and Acceleartions (Can be changed for individual commands if needed)



speed_J_fast = [3, 3]   # max is 1 (3)
acc_J_fast = [6, 3]     # max is 1 (2)

pos_init = np.array([-0.145, -0.70112, -0.04, 0, 0, 0])

if is_leaf_traj:
    pos_init = np.array([-0.15, -0.50112, -0.12, 0, 0, -1.571])


print(traj[20000, 1:7])
pos0 = rtde_c.poseTrans(pos_init, traj[20000, 1:7])
print(pos0)
rtde_c.moveL(pos0, 1, 0.5)

print("MOVING TO STARTING POSE")
time.sleep(1)


file = 'hovering_sensor.txt'
port1 = '/dev/ttyACM0'

if SENSOR_ON:
    print ("Connecting to...." + port1)
    ser = serial.Serial(port1, 115200)
    print("Arduino detected! Starting calibration!")
    t1 = threading.Thread(target=serial_reader, args=(ser, file, rtde_r))
    t1.start()


tsteps = []
accs = []
tstart = 20000

if is_leaf_traj:
    rtde_r.startFileRecording("robot_data_leaf_hopping.csv", record_variables)
    tsteps = [0, 750, 1750, 3600, 6500, 12000, 14500, 18000, 21700, 25500, 29050, 33200, 36500, 39000, 41750, 44500, 47200, 49250, 52725]
    accs = [6, 5.5, 8, 15, 4.75, 0.52, 0.8, 0.34, 1, 1, 1, 0.75, 0.65, 0.55, 1.5, 1.5, 1.5, 8, 0.1]
else:
    rtde_r.startFileRecording("robot_data_hovering.csv", record_variables)
    tsteps = [0, 750, 1600, 3000, 5600, 9250, 12800, 14000, 15550, 16750, 18100, 20000, 21300, 24250, 26250, 28250, 30900, 34500, 39500, 41000, 42250, 43750, 45500, 46900, 47600, 49100, 52797]
    accs = [6, 6, 1.75, 7.5, 3.5, 1.00, 1.05, 6.5, 7.5, 3.5, 2.6, 1.5, 1.65, 0.45, 1.4, 0.8, 0.75, 0.5, 0.32, 1.9, 2.1, 2.45, 1.1, 5, 6, 2, 0.05]




for i, t_ in enumerate(tsteps):
   t = tstart + t_

   print(t)
   pose = rtde_c.poseTrans(pos_init, traj[t, 1:7])
#    pose[:2] = pos0[:2]

   speed = 3
   acc = accs[i]   
   rtde_c.moveL(pose, speed, acc, asynchronous=False)

readData = False

if SENSOR_ON:
    t1.join()
rtde_r.stopFileRecording()

rtde_c.moveL(pos0, speed_J_fast[0], acc_J_fast[0])
rtde_c.stopScript()

### Generate file for state estimation

if SENSOR_ON:
    arm_df = pd.read_csv("robot_data_hovering.csv", sep=',', header=0)

    if is_leaf_traj: arm_df = pd.read_csv("robot_data_leaf_hopping.csv", sep=',', header=0)
    

    shift_t = arm_df['timestamp'][0]
    arm_df["timestamp"] = np.round(1000*(arm_df["timestamp"] - shift_t)).astype(int)




    l = list(q.queue)

    l_ = []
    for val in l[2:]:
        a= val.split(' ') 
        a[-1] = a[-1][:-1]
        a = [float(a_) for a_ in a]
        l_.append(a)

    sensor_df = pd.DataFrame(l_, columns=["timestamp", "rx", "ry", "rz", "tof"], dtype=float)
    sensor_df["timestamp"] = np.round(1000*(sensor_df["timestamp"] - shift_t)).astype(int)
    sensor_df['timestamp'] = (i:=sensor_df['timestamp'].astype(int)) - i % 2
    print(sensor_df)

    robobee_df =  pd.read_csv("hovering.csv", sep=',', header=None, names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw", "tx", "ty", "tz", "ft"])

    robobee_df["timestamp"] = robobee_df["timestamp"] - 2
    robobee_df = robobee_df.iloc[20000::10, :]
    robobee_df["timestamp"] = np.round(1000*robobee_df["timestamp"]).astype(int)


    experiment_df = sensor_df.set_index("timestamp").join(arm_df.set_index("timestamp")).dropna().reset_index()
    experiment_df = experiment_df.set_index("timestamp").join(robobee_df.set_index("timestamp")).dropna().reset_index()
    experiment_df["timestamp"] = 1e-3*experiment_df["timestamp"]


    reset_columns = ["rx", "ry", "rz", "x", "y", "z", "roll", "pitch", "yaw", "actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2", "actual_TCP_pose_3", "actual_TCP_pose_4", "actual_TCP_pose_5"]
    for col in reset_columns:
        experiment_df[col] = experiment_df[col] - experiment_df[col][0]

    print(sensor_df.shape)
    print(experiment_df.shape)
    experiment_df.to_csv(output_file, index=False)  