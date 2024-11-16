"""Dead time approximation based on smith predictor."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "../data/motor-calibration.jsonl"

df = pd.read_json(filename, orient="records", lines=True)
print(df.head())

L_time = df["timeNow"].max() - df["timeNow"].min()
T_sample = 0.01  # [sec]
tau = 0.01
km = 1.0  #10.0
L = 0.3

a1 = -12/(tau*L**2)
a2 = -12*(tau+L/2)/(tau*L**2)
a3 = -(6*tau*L+L**2)/(tau*L**2)
b = 12*km/(tau*L**2)

time_simulated = df["timeNow"].to_numpy()
goal_angle_speed_simulated = 3.1415 * 3.0 * np.ones(len(df))
goal_angle_speed_simulated[time_simulated < 3.00] = 0.0

x1 = 0
x2 = 0
x3 = 0
x1s = []
x2s = []
x3s = []
for t, u in zip(time_simulated, goal_angle_speed_simulated):
    dx1 = 1.0 * x2
    dx2 = 1.0 * x3
    dx3 = a1 * x1 + a2 * x2 + a3 * x3 + b * u
    x1 += dx1 * T_sample
    x2 += dx2 * T_sample
    x3 += dx3 * T_sample
    x1s.append(x1)
    x2s.append(x2)
    x3s.append(x3)

plt.figure()
plt.plot(df["timeNow"], df["goalAngleSpeed"], label="goal")
plt.plot(df["timeNow"], df["angleSpeedA"], label="angleSpeedA")
plt.plot(df["timeNow"], df["angleSpeedB"], label="angleSpeedB")
plt.plot(df["timeNow"], x1s, label="simulatedAngleSpeed")
# plt.plot(df["timeNow"], x2s, label="simulatedAngleAccel")
# plt.plot(df["timeNow"], x3s, label="simulatedAngleAccel")
plt.legend()
plt.show()
