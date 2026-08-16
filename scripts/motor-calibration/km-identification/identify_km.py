"""
Identify the actuator parameters of the inverted pendulum from measure_km.ino.

What this script produces
-------------------------
1. G   : the motor DC gain, in [rad/s per unit duty], per wheel and per
         direction, together with the dead zone (the duty below which the
         wheel does not turn at all).
2. L_d : the pure dead time, and tau_m, the first-order time constant,
         fitted from the step segments.
3. km  : the motor gain as the design model defines it, i.e. the steady
         gain from the LQR command u to the wheel speed omega, obtained by
         combining G with the lower loop that the balance sketch runs.

Why km is not a single number
-----------------------------
The balance sketch converts u into a duty with

    duty[k] = kp * e[k] + ki * I[k],   I[k] = e[k] * Ts + rho * I[k-1]

(with e = u because the wheel-speed feedback is inactive; see
docs/specs/hardware/README.md).  That is a lead-lag, not a constant:

    duty/u  ->  kp + ki * Ts/(1-rho)   at DC          (= 0.065 for wheel A)
    duty/u  ->  kp                     well above the
                                       leak corner    (= 0.025 for wheel A)

The leak corner sits at (1-rho)/Ts = 1.0 rad/s, while balancing happens
around the unstable pole 2*pi/T_p = 8.4 rad/s.  The script therefore
reports km at DC and at the unstable-pole frequency, and it is the latter
that matters for the balance design.

Usage
-----
    # analyse a capture
    uv run python scripts/motor-calibration/km-identification/identify_km.py \
        data/km-measurement.csv

    # capture from the board and analyse in one go (needs pyserial)
    uv run --with pyserial python \
        scripts/motor-calibration/km-identification/identify_km.py \
        --port /dev/ttyUSB0 --save data/km-measurement.csv

    # skip the plot
    ... --no-plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ============================================================
# Constants that must mirror the balance sketch
# ============================================================

TS = 0.010          # control period [s]
RHO = 0.99          # leaky-integrator pole

KP = {"A": 0.025, "B": 0.030}
KI = {"A": 0.040, "B": 0.050}

T_PENDULUM = 0.75   # [s]; only used to report the balance-relevant frequency

# A level is considered settled after this fraction of its hold time.
SETTLE_FRACTION = 0.45

# Levels whose |duty| is at least this are ignored for the DC-gain fit,
# because the supply sags and the motor saturates there.
MAX_FIT_DUTY = 0.95

# A wheel is considered "not moving" below this speed [rad/s].
STALL_SPEED = 0.3


# ============================================================
# Capture
# ============================================================

def capture(port: str, baud: int, save: str | None) -> list[str]:
    try:
        import serial
    except ImportError:
        sys.exit(
            "pyserial is required for --port.\n"
            "Run with:  uv run --with pyserial python <this script> --port ..."
        )

    lines: list[str] = []

    with serial.Serial(port, baud, timeout=5) as ser:
        print(f"capturing from {port} ...", file=sys.stderr)
        started = False

        while True:
            raw = ser.readline()

            if not raw:
                print("serial timed out", file=sys.stderr)
                break

            text = raw.decode("utf-8", errors="replace").rstrip()
            lines.append(text)

            if text.startswith("#"):
                print(text, file=sys.stderr)

            if "# BEGIN" in text:
                started = True

            if started and "# END" in text:
                break

    if save:
        with open(save, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"saved to {save}", file=sys.stderr)

    return lines


def load(lines: list[str]) -> pd.DataFrame:
    rows = []

    for text in lines:
        text = text.strip()

        if not text or text.startswith("#"):
            continue

        parts = text.split(",")

        if len(parts) != 4:
            continue

        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue

    if not rows:
        sys.exit("no data rows found")

    frame = pd.DataFrame(rows, columns=["t_ms", "duty", "phiA", "phiB"])
    frame["t"] = frame["t_ms"] * 1e-3
    return frame


# ============================================================
# Segmentation
# ============================================================

@dataclass
class Segment:
    duty: float
    t: np.ndarray
    phi: dict[str, np.ndarray]


def segments(frame: pd.DataFrame) -> list[Segment]:
    """Split the capture wherever the commanded duty changes."""
    duty = frame["duty"].to_numpy()
    edges = np.flatnonzero(np.diff(duty) != 0.0) + 1
    bounds = np.concatenate(([0], edges, [len(duty)]))

    out = []

    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop - start < 5:
            continue

        out.append(
            Segment(
                duty=float(duty[start]),
                t=frame["t"].to_numpy()[start:stop],
                phi={
                    "A": frame["phiA"].to_numpy()[start:stop],
                    "B": frame["phiB"].to_numpy()[start:stop],
                },
            )
        )

    return out


def steady_speed(seg: Segment, wheel: str) -> float:
    """
    Steady-state wheel speed [rad/s] over the settled part of a segment.

    The speed is the slope of a straight line fitted to the wheel ANGLE.
    Fitting the angle rather than differentiating it keeps the coarse
    encoder resolution (0.187 rad/count) from dominating the result.
    """
    n = len(seg.t)
    lo = int(n * SETTLE_FRACTION)

    t = seg.t[lo:]
    phi = seg.phi[wheel][lo:]

    if len(t) < 5:
        return float("nan")

    slope, _ = np.polyfit(t - t[0], phi, 1)
    return float(slope)


# ============================================================
# DC gain and dead zone
# ============================================================

def fit_dc_gain(levels: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Fit  omega = G * (|duty| - d0)  for |duty| > d0, omega = 0 otherwise.

    Returns (G, d0).  `levels` holds (|duty|, |omega|) pairs.
    """
    duty = np.array([d for d, _ in levels])
    speed = np.array([w for _, w in levels])

    usable = (duty <= MAX_FIT_DUTY) & np.isfinite(speed)

    if usable.sum() < 3:
        return float("nan"), float("nan")

    duty = duty[usable]
    speed = speed[usable]

    def residual(params):
        gain, dead = params
        model = gain * np.clip(duty - dead, 0.0, None)
        return model - speed

    guess = [speed.max() / max(duty.max(), 1e-6), 0.05]
    result = least_squares(residual, guess, bounds=([0.0, 0.0], [np.inf, 0.9]))

    return float(result.x[0]), float(result.x[1])


# ============================================================
# Dead time and motor time constant
# ============================================================

def fit_step(seg: Segment, wheel: str, gain: float, dead: float):
    """
    Fit the dead time L and the time constant tau on a rising step.

    The wheel ANGLE is fitted, not the speed:

        omega(t) = omega_ss * (1 - exp(-(t - L)/tau))   for t >= L
        phi(t)   = phi0 + omega_ss * ((t - L) + tau * (exp(-(t-L)/tau) - 1))

    Differentiating the encoder would swamp tau (0.07 s) in quantization
    noise, so the integral form is used instead.
    """
    t = seg.t - seg.t[0]
    phi = seg.phi[wheel] - seg.phi[wheel][0]

    duty = abs(seg.duty)
    direction = np.sign(seg.duty)

    if not np.isfinite(gain) or duty <= dead:
        return None

    # A wheel that never turned carries no information about L or tau, and
    # least_squares would happily hand back the initial guess.
    if abs(phi[-1] - phi[0]) < 3.0 * STALL_SPEED * (t[-1] - t[0]):
        return None

    omega_ss = direction * gain * (duty - dead)

    def model(params):
        lag, tau, phi0 = params
        u = np.clip(t - lag, 0.0, None)
        return phi0 + omega_ss * (u + tau * (np.exp(-u / tau) - 1.0))

    def residual(params):
        return model(params) - phi

    result = least_squares(
        residual,
        [0.10, 0.07, 0.0],
        bounds=([0.0, 0.005, -1.0], [0.5, 1.0, 1.0]),
    )

    lag, tau, _ = result.x
    return float(lag), float(tau), model(result.x), t, phi


# ============================================================
# Lower loop:  duty / u
# ============================================================

def lower_loop_gain(wheel: str, freq_rad_s: float) -> complex:
    """
    duty(z)/u(z) = kp + ki * Ts / (1 - rho z^-1), evaluated at z = e^{jwTs}.
    """
    z = np.exp(1j * freq_rad_s * TS)
    return KP[wheel] + KI[wheel] * TS / (1.0 - RHO / z)


# ============================================================
# Report
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="capture file from measure_km.ino")
    parser.add_argument("--port", help="serial port to capture from")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--save", help="where to write the capture")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.port:
        lines = capture(args.port, args.baud, args.save)
    elif args.csv:
        with open(args.csv) as handle:
            lines = handle.read().splitlines()
    else:
        parser.error("give a capture file or --port")

    frame = load(lines)
    segs = segments(frame)

    print("=" * 70)
    print("CAPTURE")
    print("=" * 70)
    print(f"samples          : {len(frame)}")
    print(f"duration         : {frame['t'].iloc[-1] - frame['t'].iloc[0]:.2f} s")
    print(f"duty segments    : {len(segs)}")

    results = {}

    for wheel in ("A", "B"):
        print()
        print("=" * 70)
        print(f"WHEEL {wheel}")
        print("=" * 70)
        print(f"{'duty':>8} {'omega [rad/s]':>14} {'omega/duty':>12}")

        forward, reverse = [], []

        for seg in segs:
            if seg.duty == 0.0:
                continue

            speed = steady_speed(seg, wheel)

            if not np.isfinite(speed):
                continue

            ratio = speed / seg.duty if seg.duty else float("nan")
            print(f"{seg.duty:8.2f} {speed:14.3f} {ratio:12.2f}")

            pair = (abs(seg.duty), abs(speed))

            if seg.duty > 0:
                forward.append(pair)
            else:
                reverse.append(pair)

        gain_f, dead_f = fit_dc_gain(forward)
        gain_r, dead_r = fit_dc_gain(reverse)

        print()
        print(f"  forward : G = {gain_f:7.2f} rad/s per duty, dead zone = {dead_f:.3f}")
        print(f"  reverse : G = {gain_r:7.2f} rad/s per duty, dead zone = {dead_r:.3f}")

        # ---- stuck-PWM detection ----
        moving = [(d, w) for d, w in forward + reverse if w > STALL_SPEED]
        stalled = [(d, w) for d, w in forward + reverse if w <= STALL_SPEED]

        if moving and all(d > 0.99 for d, _ in moving) and stalled:
            print()
            print("  *** WHEEL " + wheel + " ONLY TURNS AT |duty| = 1.00 ***")
            print("  Every intermediate duty produced no motion at all.  That is the")
            print("  signature of a broken PWM output: analogWrite(pin, 255) is the one")
            print("  value the Arduino core turns into digitalWrite(HIGH), so a pin whose")
            print("  timer has been repurposed still drives at full duty and nowhere else.")
            print("  On the Uno, D3 and D11 belong to Timer2, and MsTimer2 takes Timer2")
            print("  over.  Drop MsTimer2 (schedule the cycle from micros() instead) and")
            print("  measure again.  See docs/specs/hardware/README.md.")

        gains = [g for g in (gain_f, gain_r) if np.isfinite(g)]
        gain = float(np.mean(gains)) if gains else float("nan")
        dead = float(np.nanmean([dead_f, dead_r]))

        if np.isfinite(gain_f) and np.isfinite(gain_r):
            skew = abs(gain_f - gain_r) / gain * 100.0
            print(f"  mean    : G = {gain:7.2f}   forward/reverse mismatch {skew:.1f}%")

            if skew > 20.0:
                print("  NOTE: the two directions differ by more than 20%.")
                print("        D3 and D11 are Timer2 pins and MsTimer2 owns Timer2,")
                print("        so one direction per motor may not be driven with a")
                print("        proper PWM waveform.  See docs/specs/hardware/README.md.")

        # ---- steps ----
        step_fits = []

        for index, seg in enumerate(segs):
            if seg.duty == 0.0 or index == 0:
                continue

            if segs[index - 1].duty != 0.0 or abs(seg.duty) > MAX_FIT_DUTY:
                continue

            fit = fit_step(seg, wheel, gain, dead)

            if fit:
                step_fits.append(fit)

        if not step_fits:
            print()
            print("  no usable step segment (the wheel did not move on any rising edge)")

        if step_fits:
            lags = [f[0] for f in step_fits]
            taus = [f[1] for f in step_fits]
            print()
            print(f"  step fits ({len(step_fits)} rising edges):")
            print(f"    dead time L_d = {np.mean(lags):.4f} s  (spread {np.std(lags):.4f})")
            print(f"    time const tau_m = {np.mean(taus):.4f} s  (spread {np.std(taus):.4f})")

        results[wheel] = dict(
            gain=gain, dead=dead, forward=forward, reverse=reverse, steps=step_fits
        )

    # ------------------------------------------------------------
    # km for the design model
    # ------------------------------------------------------------
    freq = 2.0 * np.pi / T_PENDULUM

    print()
    print("=" * 70)
    print("km FOR THE DESIGN MODEL")
    print("=" * 70)
    print(f"leak corner            : {(1 - RHO) / TS:.2f} rad/s")
    print(f"unstable pole 2pi/T_p  : {freq:.2f} rad/s   (T_p = {T_PENDULUM} s)")
    print()
    print(f"{'wheel':>6} {'G':>8} {'duty/u DC':>11} {'duty/u @wb':>11} {'km DC':>8} {'km @wb':>8}")

    for wheel in ("A", "B"):
        gain = results[wheel]["gain"]

        if not np.isfinite(gain):
            continue

        dc = abs(lower_loop_gain(wheel, 0.0))
        hf = abs(lower_loop_gain(wheel, freq))
        print(f"{wheel:>6} {gain:8.2f} {dc:11.4f} {hf:11.4f} {gain * dc:8.2f} {gain * hf:8.2f}")

    print()
    print("Use the '@wb' column for the balance design: that is the band the")
    print("controller actually works in.  Put it into design_best_R14_controller.py")
    print("as the motor gain km (omega_dot = (-omega + km*z1)/tau_m and the matching")
    print("term in theta_ddot), then regenerate the gains.")

    if not args.no_plot:
        plot(results, frame)


def plot(results, frame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping the plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for column, wheel in enumerate(("A", "B")):
        data = results[wheel]

        ax = axes[0][column]

        for pairs, label in ((data["forward"], "forward"), (data["reverse"], "reverse")):
            if pairs:
                duty = [p[0] for p in pairs]
                speed = [p[1] for p in pairs]
                ax.plot(duty, speed, "o", label=label)

        gain, dead = data["gain"], data["dead"]

        if np.isfinite(gain):
            grid = np.linspace(0, 1, 50)
            ax.plot(
                grid,
                gain * np.clip(grid - dead, 0, None),
                "k--",
                label=f"fit G={gain:.1f}, dead={dead:.3f}",
            )

        ax.set_title(f"wheel {wheel}: duty vs steady speed")
        ax.set_xlabel("|duty|")
        ax.set_ylabel("|omega| [rad/s]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1][column]

        for fit in data["steps"]:
            lag, tau, model, t, phi = fit
            ax.plot(t, phi, ".", markersize=3, alpha=0.5)
            ax.plot(t, model, "-", linewidth=1, label=f"L={lag:.3f}, tau={tau:.3f}")

        ax.set_title(f"wheel {wheel}: step response (angle)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("phi [rad]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
