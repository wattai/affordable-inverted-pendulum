"""Dead time approximation based on multi layered lagged elements."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "../data/motor-calibration.jsonl"

df = pd.read_json(filename, orient="records", lines=True)
print(df.head())

L_time = df["timeNow"].max() - df["timeNow"].min()
T_sample = 0.01  # [sec]

tau = 0.07
km = 1.0  #10.0
L = 0.10

N = 6  # 多段遅延要素の数
theta = L/N

x = np.array(
    [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]
)
A = np.array(
    [
        [-1/tau, km/tau, 0, 0, 0, 0, 0],        
        [0, -1/theta, 1/theta, 0, 0, 0, 0],
        [0, 0, -1/theta, 1/theta, 0, 0, 0],
        [0, 0, 0, -1/theta, 1/theta, 0, 0],
        [0, 0, 0, 0, -1/theta, 1/theta, 0],
        [0, 0, 0, 0, 0, -1/theta, 1/theta],
        [0, 0, 0, 0, 0, 0, -1/theta],
    ]
)
B = np.array(
    [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1/theta],
    ]
)

time_simulated = df["timeNow"].to_numpy()
goal_angle_speed_simulated = 3.1415 * 3.0 * np.ones(len(df))
goal_angle_speed_simulated[time_simulated < 3.00] = 0.0

xs = []
for t, u in zip(time_simulated, goal_angle_speed_simulated):
    dx = A @ x + B @ np.array([[u]])
    x = x + dx * T_sample
    xs.append(x)

plt.figure()
plt.plot(df["timeNow"], df["goalAngleSpeed"], label="goal")
plt.plot(df["timeNow"], df["angleSpeedA"], label="angleSpeedA")
plt.plot(df["timeNow"], df["angleSpeedB"], label="angleSpeedB")
plt.plot(df["timeNow"], np.array(xs).squeeze()[:, :], label="simulatedAngleSpeed")
# plt.plot(df["timeNow"], x2s, label="simulatedAngleAccel")
# plt.plot(df["timeNow"], x3s, label="simulatedAngleAccel")
plt.legend()
plt.show()
