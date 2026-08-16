"""
Reproduce the controller design that produced the current best Arduino behavior.

Best real-robot configuration discussed in this project:
    - Ts = 0.010 s
    - wheel radius = 0.027 m
    - body mass = 0.350 kg
    - CoM height = 0.060 m
    - measured pendulum period = 0.75 s
    - motor first-order time constant = 0.070 s
    - six-stage delay approximation = 0.110 s total
    - LQR Q = diag(10, 1, 0, 1, 0, 0, 0, 0, 0, 0)
    - LQR R = 1.4
    - Arduino actually applies only the first four LQR gains:
          theta, theta_dot, wheel_angle, wheel_speed
      and DOES NOT directly feed back z1..z6.
    - left/right synchronization gain = 1.20

State order:
    x = [
        theta,
        theta_dot,
        phi,
        omega,
        z1, z2, z3, z4, z5, z6
    ]

where
    theta      : body angle [rad]
    theta_dot  : body angular rate [rad/s]
    phi        : average wheel angle [rad]
    omega      : average wheel angular rate [rad/s]
    z1..z6     : six cascaded first-order states approximating input delay

This script deliberately reproduces the numerical procedure that generated
the R=1.4 gains used in the Arduino code:

    1. Build the continuous 10-state model from physical parameters.
    2. Discretize it with zero-order hold at 10 ms.
    3. Round A_d and B_d to 6 decimal places, matching the matrices copied
       into the Arduino implementation.
    4. Solve the DISCRETE Riccati equation.
    5. Compute

         K = (R + B^T P B)^(-1) B^T P A

       for u[k] = -K x[k].

The 6-decimal matrix rounding is important.  If the full-precision matrices
are used instead, the R=1.4 gain differs slightly from the values currently
embedded in the successful Arduino program.

Requirements:
    pip install numpy scipy

Optional:
    pip install matplotlib

The script also prints an Arduino-ready parameter block.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals, solve_discrete_are
from scipy.signal import cont2discrete


# ============================================================
# 1. Physical / timing parameters
# ============================================================

TS = 0.010                      # [s]

R_WHEEL = 0.027                 # [m]
G = 9.8                         # [m/s^2]
M_BODY = 0.350                  # [kg]
H_BODY = 0.060                  # [m]

T_PENDULUM = 0.75               # [s]
TAU_MOTOR = 0.070               # [s]

# The 2026-08-15 design used 0.100 s here.  The measured pure delay is
# closer to 0.110 s, so this configuration uses the measured value.
#
# NOTE on T_PENDULUM:
# A video measurement using a string tied to the axle suggested ~0.45 s,
# but that setup is a two-degree-of-freedom pendulum (the string can swing
# too), so it does not measure rotation about a fixed axle.  0.45 s is in
# fact impossible for this body: a rigid body with its CoM 0.060 m from the
# pivot cannot swing faster than 2*pi*sqrt(h/g) = 0.492 s.  The value below
# is the nail-through-the-wheel-axis measurement, which is the single-pivot
# compound pendulum the model actually assumes.
DELAY_SEC = 0.110
DELAY_STAGES = 6

# ============================================================
# 2. LQR weights
# ============================================================

Q_THETA = 10.0
Q_THETA_DOT = 1.0
Q_WHEEL_ANGLE = 0.0
Q_WHEEL_SPEED = 1.0

# Current best real-robot setting.
R_INPUT = 1.4

# True reproduces the matrices actually used to calculate the successful
# R=1.4 Arduino gains.
USE_ARDUINO_6_DECIMAL_MATRICES = True

# The successful Arduino controller did NOT directly apply z1..z6 gains.
# Leave False to reproduce that implementation.
APPLY_DELAY_STATE_FEEDBACK_TO_ARDUINO = False

# Empirical lower-level parameters from the successful Arduino configuration.
K_WHEEL_SYNC = 1.20
KP_A = 0.025
KP_B = 0.030
KI_A = 0.040
KI_B = 0.050


# ============================================================
# Utility
# ============================================================

def print_matrix(name: str, matrix: np.ndarray, precision: int = 9) -> None:
    print(f"\n{name} =")
    print(
        np.array2string(
            np.asarray(matrix),
            precision=precision,
            suppress_small=True,
            floatmode="fixed",
        )
    )


def discrete_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the discrete-time infinite-horizon LQR problem

        x[k+1] = A x[k] + B u[k]
        u[k]   = -K x[k]

    minimizing

        J = sum(x[k]^T Q x[k] + u[k]^T R u[k]).

    Correct discrete-time gain:
        K = (R + B^T P B)^(-1) B^T P A
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    poles = eigvals(A - B @ K)
    return K, P, poles


# ============================================================
# 3. Effective body inertia
# ============================================================

I_EFF = (
    T_PENDULUM**2
    * M_BODY
    * G
    * H_BODY
    / (4.0 * np.pi**2)
)

print("=" * 78)
print("PHYSICAL PARAMETERS")
print("=" * 78)

print(f"Ts                 = {TS:.6f} s")
print(f"wheel radius       = {R_WHEEL:.6f} m")
print(f"body mass          = {M_BODY:.6f} kg")
print(f"CoM height         = {H_BODY:.6f} m")
print(f"pendulum period    = {T_PENDULUM:.6f} s")
print(f"effective inertia  = {I_EFF:.9f} kg m^2")
print(f"motor tau          = {TAU_MOTOR:.6f} s")
print(f"delay model        = {DELAY_SEC:.6f} s / {DELAY_STAGES} stages")


# ============================================================
# 4. Continuous-time 10-state plant
# ============================================================

N = 4 + DELAY_STAGES

A_c = np.zeros((N, N), dtype=float)
B_c = np.zeros((N, 1), dtype=float)

# Body dynamics
a_theta = M_BODY * G * H_BODY / I_EFF

# Coupling between wheel motion and body angular acceleration.
a_body_wheel = (
    M_BODY
    * H_BODY
    * R_WHEEL
    / (I_EFF * TAU_MOTOR)
)

# Six cascaded first-order delay states.
delay_rate = DELAY_STAGES / DELAY_SEC

# theta_dot
A_c[0, 1] = 1.0

# theta_ddot
#
# theta_ddot =
#       a_theta * theta
#     + a_body_wheel * omega
#     - a_body_wheel * z1
#
A_c[1, 0] = a_theta
A_c[1, 3] = a_body_wheel
A_c[1, 4] = -a_body_wheel

# phi_dot = omega
A_c[2, 3] = 1.0

# Motor:
# omega_dot = -(1/tau)*omega + (1/tau)*z1
A_c[3, 3] = -1.0 / TAU_MOTOR
A_c[3, 4] = +1.0 / TAU_MOTOR

# Delay chain:
#
# z1_dot = lambda(-z1 + z2)
# z2_dot = lambda(-z2 + z3)
# ...
# z6_dot = lambda(-z6 + u)
#
for row in range(4, N):
    A_c[row, row] = -delay_rate

for row in range(4, N - 1):
    A_c[row, row + 1] = delay_rate

B_c[N - 1, 0] = delay_rate

print("\nDerived coefficients:")
print(f"a_theta           = {a_theta:.9f}")
print(f"a_body_wheel      = {a_body_wheel:.9f}")
print(f"delay rate lambda = {delay_rate:.9f} 1/s")
print(f"stage tau         = {1.0 / delay_rate:.9f} s")

print_matrix("A_c", A_c)
print_matrix("B_c", B_c)


# ============================================================
# 5. ZOH discretization
# ============================================================

C_dummy = np.eye(N)
D_dummy = np.zeros((N, 1))

A_d_exact, B_d_exact, _, _, _ = cont2discrete(
    (A_c, B_c, C_dummy, D_dummy),
    TS,
    method="zoh",
)

# The successful R=1.4 design was generated using the six-decimal matrices
# that mirror the Arduino source.
A_d_arduino = np.round(A_d_exact, 6)
B_d_arduino = np.round(B_d_exact, 6)

if USE_ARDUINO_6_DECIMAL_MATRICES:
    A_design = A_d_arduino
    B_design = B_d_arduino
    matrix_source = "6-decimal Arduino matrices"
else:
    A_design = A_d_exact
    B_design = B_d_exact
    matrix_source = "full-precision discretized matrices"

print("\n" + "=" * 78)
print("DISCRETE MODEL")
print("=" * 78)
print(f"LQR matrix source: {matrix_source}")

print_matrix("A_d_exact", A_d_exact)
print_matrix("B_d_exact", B_d_exact)

print_matrix("A_d_used_for_design", A_design, precision=6)
print_matrix("B_d_used_for_design", B_design, precision=6)


# ============================================================
# 6. LQR
# ============================================================

Q = np.diag(
    [
        Q_THETA,
        Q_THETA_DOT,
        Q_WHEEL_ANGLE,
        Q_WHEEL_SPEED,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)

R = np.array([[R_INPUT]], dtype=float)

K_full, P, poles_full = discrete_lqr(
    A_design,
    B_design,
    Q,
    R,
)

print("\n" + "=" * 78)
print("FULL 10-STATE DISCRETE LQR")
print("=" * 78)

print_matrix("Q", Q)
print_matrix("R", R)
print_matrix("P", P)
print_matrix("K_full", K_full)

print("\nState order:")
print("[theta, theta_dot, phi, omega, z1, z2, z3, z4, z5, z6]")

print("\nFull-LQR closed-loop poles:")
for pole in poles_full:
    print(
        f"  {pole.real:+.9f}{pole.imag:+.9f}j"
        f"   |p|={abs(pole):.9f}"
    )

print(
    f"max |closed-loop pole| = "
    f"{np.max(np.abs(poles_full)):.9f}"
)


# ============================================================
# 7. Reproduce the actual successful Arduino feedback structure
# ============================================================

K_arduino = K_full.copy()

if not APPLY_DELAY_STATE_FEEDBACK_TO_ARDUINO:
    # This is exactly the empirical structure that performed best:
    # use the first four LQR gains and do not directly add z feedback.
    K_arduino[0, 4:] = 0.0

print("\n" + "=" * 78)
print("ACTUAL ARDUINO FEEDBACK STRUCTURE")
print("=" * 78)

if APPLY_DELAY_STATE_FEEDBACK_TO_ARDUINO:
    print("Direct z1..z6 feedback: ENABLED")
else:
    print("Direct z1..z6 feedback: DISABLED  <-- reproduces best behavior")

print_matrix("K_arduino_applied", K_arduino)

print(
    "\nControl law used by the successful Arduino structure:\n"
    "    u = -K_theta*theta_hat\n"
    "        -K_theta_dot*theta_dot_hat\n"
    "        -K_phi*phi_hat\n"
    "        -K_omega*omega_hat\n"
)

if APPLY_DELAY_STATE_FEEDBACK_TO_ARDUINO:
    print("        -K_z1*z1_hat - ... -K_z6*z6_hat")


# ============================================================
# 8. Numerical reproduction check for the known R=1.4 values
# ============================================================

# Expected R=1.4 gain for THIS configuration
# (T_pendulum = 0.75 s, delay = 0.110 s, Q_wheel_angle = 0).
EXPECTED_R14_FULL = np.array(
    [[
        -359.828687,
        -42.957907,
        +0.000000,
        -5.218460,
        +0.703306,
        +0.611328,
        +0.529673,
        +0.457319,
        +0.392693,
        +0.334826,
    ]]
)

print("\n" + "=" * 78)
print("REPRODUCTION CHECK")
print("=" * 78)

if (
    abs(R_INPUT - 1.4) < 1e-12
    and USE_ARDUINO_6_DECIMAL_MATRICES
):
    max_gain_error = np.max(
        np.abs(K_full - EXPECTED_R14_FULL)
    )

    print(
        "max |K_calculated - K_expected_R14| = "
        f"{max_gain_error:.3e}"
    )

    if max_gain_error < 1e-6:
        print("PASS: the successful R=1.4 gain has been reproduced.")
    else:
        print("WARNING: gain does not match the expected R=1.4 design.")
else:
    print(
        "R or matrix precision differs from the successful R=1.4 setup, "
        "so the fixed reproduction check was skipped."
    )


# ============================================================
# 9. Important distinction: full LQR vs empirical Arduino controller
# ============================================================

poles_applied_nominal = eigvals(
    A_design - B_design @ K_arduino
)

print("\n" + "=" * 78)
print("NOMINAL-PLANT CHECK OF THE APPLIED ARDUINO GAIN")
print("=" * 78)

print(
    "max |pole| with K_arduino_applied = "
    f"{np.max(np.abs(poles_applied_nominal)):.9f}"
)

if np.max(np.abs(poles_applied_nominal)) >= 1.0:
    print(
        "\nNOTE:\n"
        "The four-state-only gain is NOT the exact full-state LQR solution "
        "for this standalone 10-state nominal plant; when z feedback is "
        "manually omitted, this simple nominal closed-loop pole check can "
        "show instability.\n\n"
        "That is not a contradiction with the real robot result.  The actual "
        "Arduino system also contains the observer, the lower motor/PWM loop, "
        "encoder behavior, saturation, and modeling mismatch.  Experimentally, "
        "the four-state structure was much better than directly feeding back "
        "z1..z6.  This script therefore reproduces both:\n"
        "  (a) the theoretical full LQR calculation, and\n"
        "  (b) the empirically successful Arduino gain selection.\n"
    )


# ============================================================
# 10. Arduino-ready constants
# ============================================================

print("\n" + "=" * 78)
print("ARDUINO-READY PARAMETER BLOCK")
print("=" * 78)

names = [
    "K_THETA",
    "K_THETA_DOT",
    "K_WHEEL_ANGLE",
    "K_WHEEL_SPEED",
    "K_Z1",
    "K_Z2",
    "K_Z3",
    "K_Z4",
    "K_Z5",
    "K_Z6",
]

for name, value in zip(names, K_arduino[0]):
    print(f"const float {name:<14} = {value:+.9f}f;")

print()
print(f"const float K_WHEEL_SYNC = {K_WHEEL_SYNC:.6f}f;")
print(f"const float KP_A         = {KP_A:.6f}f;")
print(f"const float KP_B         = {KP_B:.6f}f;")
print(f"const float KI_A         = {KI_A:.6f}f;")
print(f"const float KI_B         = {KI_B:.6f}f;")


# ============================================================
# 11. Optional quick comparison over several R values
# ============================================================

print("\n" + "=" * 78)
print("OPTIONAL R SWEEP")
print("=" * 78)

for r_value in [1.0, 1.2, 1.4, 1.6, 1.8]:
    K_test, _, _ = discrete_lqr(
        A_design,
        B_design,
        Q,
        np.array([[r_value]], dtype=float),
    )

    print(
        f"R={r_value:>3.1f} : "
        f"Ktheta={K_test[0,0]:>11.6f}, "
        f"KthetaDot={K_test[0,1]:>10.6f}, "
        f"Kphi={K_test[0,2]:>9.6f}, "
        f"Komega={K_test[0,3]:>9.6f}"
    )
