import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from numpy.linalg import inv
from scipy.signal import cont2discrete
np.set_printoptions(precision=6, suppress=True)

r_wheel = 3.0 * 0.01  # wheel radius [m]
g = 9.8  # gravity acceleration [m/s^2]
m_body = 180 * 0.001  # body mass [kg]
h_body = 9.0 * 0.01  # body height between wheel axis and body mass center [m]

# L_body = 14.5 * 0.01  # length of the body from wheel axis to the top of the body [m]
m_wheel = 4.5 * 0.001  # [kg]
tau_motor = 0.05  # [s]

# I_body = 1 / 12 * m_body * (pow(L_body, 2) + pow(0.02, 2)) # + pow(h_body, 2))  # body moment of inertia [?]
# I_eff = I_body + m_body * pow(h_body, 2)
T_pendulum = 0.7  #  0.35  # 振り子周期 [s]
I_eff = pow(T_pendulum, 2) * m_body * g * h_body / (4 * pow(np.pi, 2))
# I_eff = m_body * h_body ** 2
print(f"I_eff: {I_eff}")

if __name__ == "__main__":
    # システム行列
    A = np.array(
        [
            [0, 1, 0],  # body angle
            # [-m_body*g*h_body / I_eff, 0, m_body*h_body*r_wheel / (I_eff)],  # body angle speed
            [-m_body*g*h_body / I_eff, 0, m_body*h_body*r_wheel / (I_eff * tau_motor)],  # body angle speed
            [0, 0, -1/tau_motor],  # wheel speed
        ],
    )
    print(A)
    B = np.array(
        [
            [0],
            # [-m_body*h_body*r_wheel / (I_eff * tau_motor)],
            [0],
            [1/tau_motor],
            # [0],
            # [m_body*h_body*r_wheel / (I_body + m_body * h_body**2)],
            # [1],
        ],
    )
    print(B)
    # C行列とD行列は0として定義
    C = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    # C = np.array([[1, 1, 0]])
    D = np.zeros((B.shape[0], B.shape[1]))

    # 重み行列
    Q = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )
    R = np.array([[1]])

    # リカッチ方程式を解く
    P = solve_continuous_are(A, B, Q, R)

    # フィードバックゲイン K を計算
    K = np.dot(inv(R), np.dot(B.T, P))

    print("LQRゲイン K:", K)

    # 離散化 ----------------

    # サンプリング時間
    T_s = 0.010  # 例: 10ms

    # 離散化を行う方法 2: scipyのcont2discreteを使用
    # # C行列とD行列は0として定義
    # C = np.eye(A.shape[0])
    # D = np.zeros((B.shape[0], B.shape[1]))

    # scipyのcont2discreteを使って離散化
    sys_d = cont2discrete((A, B, C, D), T_s)

    A_d2, B_d2, C_d2, D_d2, dt = sys_d

    print("\nA_d (scipy.cont2discreteによる):")
    print(A_d2)
    print("\nB_d (scipy.cont2discreteによる):")
    print(B_d2)

    P_d = solve_discrete_are(A_d2, B_d2, Q, R)
    # K_d = np.dot(inv(R), np.dot(B_d2.T, P_d))
    # フィードバックゲインの計算
    K_d = np.dot(inv(np.dot(B_d2.T, np.dot(P_d, B_d2)) + R), np.dot(B_d2.T, np.dot(P_d, A_d2)))
    print("discrete LQR gain K_d:", K_d)

    from control.matlab import dlqr

    K_d, S, E = dlqr(A_d2, B_d2, Q, R)
    print("discrete LQR gain K_d:", K_d)

    from scipy.signal import place_poles

    # A の固有値（ポール）を計算
    poles = np.linalg.eigvals(A_d2)
    print("システムのポール:", poles)

    # 望ましい極を設定
    desired_poles = [0.5, 0.6, 0.9]
    # 極配置アルゴリズムでオブザーバゲイン L を計算
    result = place_poles(A_d2.T, C_d2.T, desired_poles)
    L = result.gain_matrix.T
    print("オブザーバゲイン L:", L)

# ❯ python scripts/calc-optimal-gain.py
# I_eff: 0.001970504511594549
# [[  0.         1.         0.      ]
#  [-80.568199   0.         0.246637]
#  [  0.         0.       -20.      ]]
# [[ 0.]
#  [ 0.]
#  [20.]]
# LQRゲイン K: [[-2.72359   0.959344  0.422554]]

# A_d (scipy.cont2discreteによる):
# [[ 0.995974  0.009987  0.000012]
#  [-0.804601  0.995974  0.002232]
#  [ 0.        0.        0.818731]]

# B_d (scipy.cont2discreteによる):
# [[0.000001]
#  [0.000231]
#  [0.181269]]
# discrete LQR gain K_d: [[-2.988227  0.905523  0.365169]]
# discrete LQR gain K_d: [[-2.988227  0.905523  0.365169]]
# システムのポール: [0.995974+0.089639j 0.995974-0.089639j 0.818731+0.j      ]
# オブザーバゲイン L: [[ 0.49597   0.009054]
#  [-0.805528  0.31471 ]
#  [-0.040567 -7.96314 ]]