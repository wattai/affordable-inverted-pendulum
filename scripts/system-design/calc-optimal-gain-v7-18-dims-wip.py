from typing import Optional, Any

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from numpy.linalg import inv
from scipy.signal import cont2discrete
np.set_printoptions(precision=6, suppress=True, linewidth=250)

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

L = 0.10  # むだ時間 [sec]
N = 6  # 多段遅延要素の数
theta = L / N

# サンプリング時間
T_s = 0.010  # 例: 10ms

def check_discrete_observability(A_d, C_d):
    """
    離散時間システムの可観測性を確認します。

    Parameters:
    A_d (ndarray): 離散システムの状態行列 (n x n)
    C_d (ndarray): 離散システムの出力行列 (p x n)

    Returns:
    observable (bool): 可観測であれば True
    """
    n = A_d.shape[0]
    O = C_d
    for i in range(1, n):
        O = np.vstack((O, np.dot(C_d, np.linalg.matrix_power(A_d, i))))
    rank = np.linalg.matrix_rank(O)
    return rank == n

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

def make_delay_state_transition_matrix(ndim_delay: int, dead_time_width: float) -> np.ndarray:
    theta = dead_time_width / ndim_delay
    return -1/theta * np.eye(ndim_delay) + 1/theta * np.eye(ndim_delay, k=1)

def make_delay_control_matrix(ndim_delay: int, dead_time_width: float) -> np.ndarray:
    theta = dead_time_width / ndim_delay
    return np.block([[np.zeros([ndim_delay - 1, 1])], [1/theta * np.ones([1, 1])]])


def format_array_as_curly_braces(array: np.ndarray, decimals: Optional[int] = None) -> str:
    """
    Formats a NumPy array into a string representation with curly braces, 
    displaying each row on a new line. Optionally, controls the number of decimal 
    places for floating-point numbers.

    Args:
        array (np.ndarray): The NumPy array to format.
        decimals (Optional[int]): The number of decimal places to display for 
            floating-point numbers. If None, all digits are shown.

    Returns:
        str: A formatted string representing the array with curly braces.

    Example:
        >>> array = np.array([[1.12345, 2.6789], [4.56, 5.1]])
        >>> print(format_array_as_curly_braces(array, decimals=2))
        {
         {1.12, 2.68},
         {4.56, 5.10}
        }
    """
    # フォーマット関数の定義
    def format_value(value: Any) -> str:
        if isinstance(value, float) and decimals is not None:
            return f"{value:.{decimals}f}"  # 小数点以下の桁数を指定
        return str(value)

    # 各行を '{...}' の形式にフォーマットし、改行を加える
    formatted_rows = ',\n '.join(
        '{' + ', '.join(format_value(val) for val in row) + '}' for row in array
    )

    # 全体を '{...}' で囲む
    return '{\n ' + formatted_rows + '\n}'

if __name__ == "__main__":
    # システム行列
    a21 = m_body*g*h_body / I_eff
    a24 = m_body*h_body*r_wheel / (I_eff * tau_motor)
    a44 = -1/tau_motor
    b2 = -m_body * h_body * r_wheel * km / (I_eff * tau_motor)
    b4 = km/tau_motor
    A_core = np.array([
        [  0,     1,   0,      0,    0,      0],
        [a21,     0,   0,  a24/2,    0,  a24/2],
        [  0,     0,   0,      1,    0,      0],
        [  0,     0,   0,    a44,    0,      0],
        [  0,     0,   0,      0,    0,      1],
        [  0,     0,   0,      0,    0,    a44],
    ])
    B_core = np.array([
        [0,        0],
        [b2/2,  b2/2],
        [0,        0],
        [b4,       0],
        [0,        0],
        [0,       b4],
    ])
    D_delay = make_delay_state_transition_matrix(N, L)
    A = np.block([
        [A_core, B_core[:, [0]], np.zeros([A_core.shape[0], N - 1]), B_core[:, [1]], np.zeros([A_core.shape[0], N - 1])],
        [np.zeros([N, A_core.shape[1]]), D_delay, np.zeros([N, D_delay.shape[1]])],
        [np.zeros([N, A_core.shape[1]]), np.zeros([N, D_delay.shape[1]]), D_delay],
    ])
    print(A)
    B = np.block([
        [np.zeros([A_core.shape[0], B_core.shape[1]])],
        [make_delay_control_matrix(N, L), np.zeros([N, 1])],
        [np.zeros([N, 1]), make_delay_control_matrix(N, L)],
    ])
    print(B)
    # C行列とD行列は0として定義
    C = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # C = np.array([[1, 1, 0]])
    D = np.zeros((B.shape[0], B.shape[1]))

    # 重み行列
    Q = np.eye(B.shape[0]) * np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])
    R = np.array([
        [100, 0],
        [0, 100],
    ])

    # リカッチ方程式を解く
    print(B.shape)
    print(R.shape)
    P = solve_continuous_are(A, B, Q, R)

    # フィードバックゲイン K を計算
    K = np.dot(inv(R), np.dot(B.T, P))

    print("Continuous LQR gain K:", K)

    # 離散化 ----------------

    # 離散化を行う方法 2: scipyのcont2discreteを使用
    # scipyのcont2discreteを使って離散化
    sys_d = cont2discrete((A, B, C, D), T_s)

    A_d, B_d, C_d, D_d, dt = sys_d

    print("\nA_d (scipy.cont2discreteによる):")
    print(format_array_as_curly_braces(A_d, decimals=6))
    print("\nB_d (scipy.cont2discreteによる):")
    print(format_array_as_curly_braces(B_d, decimals=6))
    
    print(f"Observability: {check_discrete_observability(A_d, C_d)}")

    # フィードバックゲインの計算
    from control.matlab import dlqr, dlqe

    K_d, S, E = dlqr(A_d, B_d, Q, R)
    print("discrete LQR gain K_d:")
    print(format_array_as_curly_braces(K_d, decimals=6))

    # オブザーバーゲインの計算
    print(C_d.shape)
    # Q_obs = np.eye(C_d.shape[1])
    Q_obs = np.eye(C_d.shape[1]) * np.array([
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ])
    R_obs = np.eye(C_d.shape[0]) * np.array([
        1, 1, 1, 1, 1, 1,
    ])
    print(R.shape)
    G_d = np.eye(A_d.shape[0])
    L, P, E = dlqe(A_d, G_d, C_d, Q_obs, R_obs)
    # L = calculate_discrete_observer_gain(A_d, C_d, Q_obs, R_obs)
    print("オブザーバゲイン L:")
    print(format_array_as_curly_braces(L, decimals=6))
