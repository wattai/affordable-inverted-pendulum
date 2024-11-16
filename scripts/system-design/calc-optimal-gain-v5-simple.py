import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from numpy.linalg import inv
from scipy.signal import cont2discrete
np.set_printoptions(precision=6, suppress=True)

r_wheel = 2.7 * 0.01  # wheel radius [m]
g = 9.8  # gravity acceleration [m/s^2]
m_body = 250 * 0.001  # body mass [kg]
h_body = 10.5 * 0.01  # body height between wheel axis and body mass center [m]

# L_body = 14.5 * 0.01  # length of the body from wheel axis to the top of the body [m]
m_wheel = 10.0 * 0.001  # [kg]
km = 1.0  # motor gain
tau_motor = 0.07  # [s]

# I_body = 1 / 12 * m_body * (pow(L_body, 2) + pow(0.02, 2)) # + pow(h_body, 2))  # body moment of inertia [?]
# I_eff = I_body + m_body * pow(h_body, 2)
T_pendulum = 0.75  #  0.35  # 振り子周期 [s]
I_eff = pow(T_pendulum, 2) * m_body * g * h_body / (4 * pow(np.pi, 2))
# I_eff = m_body * h_body ** 2
print(f"I_eff: {I_eff}")

def calculate_discrete_observer_gain(A_d, C_d, Q, R):
    """
    離散リッカチ方程式を解いてオブザーバゲイン L を計算します。

    Parameters:
    A_d (ndarray): 離散システムの状態行列 (n x n)
    C_d (ndarray): 離散システムの出力行列 (p x n)
    Q (ndarray): 状態コスト行列 (n x n)
    R (ndarray): 出力コスト行列 (p x p)

    Returns:
    L (ndarray): オブザーバゲイン行列 (n x p)
    """
    # 離散リッカチ方程式を解く
    P = solve_discrete_are(A_d.T, C_d.T, Q, R)

    # オブザーバゲインの計算
    temp = np.dot(C_d, np.dot(P, C_d.T)) + R
    L = np.dot(np.dot(P, C_d.T), np.linalg.inv(temp))

    return L

if __name__ == "__main__":
    # システム行列
    A = np.array(
        [
            [0, 1, 0, 0],  # body angle
            [m_body*g*h_body / I_eff, 0, 0, m_body*h_body*r_wheel / (I_eff * tau_motor)],  # body angle speed
            # [-m_body*g*h_body / I_eff, 0, 0, m_body*h_body*r_wheel / (I_eff * tau_motor)],  # body angle speed
            [0, 0, 0, 1],  # wheel angle
            [0, 0, 0, -1/tau_motor],  # wheel speed
        ],
    )
    print(A)
    B = np.array(
        [
            [0],
            [-m_body * h_body * r_wheel * km / (I_eff * tau_motor)],
            [0],
            [km/tau_motor],
        ],
    )
    print(B)
    # C行列とD行列は0として定義
    C = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # C = np.array([[1, 1, 0]])
    D = np.zeros((B.shape[0], B.shape[1]))

    # 重み行列
    Q = np.array(
        [
            [10000, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
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

    # リッカチ方程式でオブザーバゲイン L を計算
    Q_obs = Q
    R_obs = np.eye(C_d2.shape[0])
    # R = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print(R.shape)
    L = calculate_discrete_observer_gain(A_d2, C_d2, Q_obs, R_obs)
    print("オブザーバゲイン L:", L)