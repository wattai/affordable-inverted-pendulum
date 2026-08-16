// ============================================================
// Motor DC-gain / dead-time identification for the inverted pendulum
//
// Purpose:
//   Measure the actuator path  duty -> wheel speed  directly, so that the
//   motor gain km, the dead zone, the first-order time constant tau_m and
//   the pure dead time L_d can be identified instead of guessed.
//
// Why drive the duty directly (and not the LQR command u):
//   The lower loop of the balance sketch is
//       duty = kp * e + ki * I,   I[k] = e * Ts + 0.99 * I[k-1]
//   which is known exactly from the source, so it does not need to be
//   measured.  Measuring duty -> omega keeps the identification clean, and
//   the u -> omega transfer is then reconstructed analytically by
//   identify_km.py.
//
// What is intentionally kept identical to the balance sketch:
//   - the pin assignment and both encoder decoders (including the
//     asymmetric 20/14 and 20/12 gear ratios)
//   - setLeftDutyCycle / setRightDutyCycle, including the "both inputs
//     high = brake" convention
//
// What is deliberately NOT here:
//   - the IMU, the observer, the LQR and the lower PI loop
//   - MsTimer2.  The first run of this sketch kept MsTimer2 for fidelity
//     with the balance sketch and proved that MsTimer2 BREAKS the PWM:
//     wheel B did not move at all for duty 0.1 .. 0.9 and only span at
//     duty = 1.00.  MsTimer2 owns Timer2, which drives D3 and D11, so
//     analogWrite() on those pins produces no usable PWM -- except for
//     the value 255, which the Arduino core turns into digitalWrite(HIGH).
//     The 10 ms cycle is therefore scheduled from micros() here, leaving
//     every timer alone so all four motor pins really do PWM.
//
// SAFETY:
//   Put the robot on a stand with both wheels free.  It will spin the
//   wheels up to full duty in both directions for about 40 seconds.
//
// Output (115200 baud, one line per 10 ms control cycle):
//   t_ms,duty,phiA,phiB
//   phiA / phiB are wheel angles in rad, already including the gear ratio.
//   Speeds are NOT computed here: the encoder resolution is coarse
//   (0.187 rad/count), so differentiating at 100 Hz would be dominated by
//   quantization.  identify_km.py fits the angle instead.
//
// Markers:
//   "# BEGIN" before the first sample, "# END" after the last one.
//   Everything starting with '#' is a comment line for the parser.
// ============================================================

#define sign(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : 0))

// ------------------------------------------------------------
// Pins (identical to the balance sketch)
// ------------------------------------------------------------
const int MOTOR_A_PIN_IN1 = 3;
const int MOTOR_A_PIN_IN2 = 9;
const int MOTOR_B_PIN_IN1 = 10;
const int MOTOR_B_PIN_IN2 = 11;

const int ROTARY_ENCODER_A_PIN_1 = 4;
const int ROTARY_ENCODER_A_PIN_2 = 2;
const int ROTARY_ENCODER_B_PIN_1 = 8;
const int ROTARY_ENCODER_B_PIN_2 = 7;

// ------------------------------------------------------------
// Timing (identical to the balance sketch)
// ------------------------------------------------------------
const unsigned long CONTROL_PERIOD_US = 10000UL;
const float ENCODER_POLL_SEC = 0.0001f;

unsigned long nextControlUs = 0;

float timeNow = 0.0f;
float timePrev = 0.0f;

// ------------------------------------------------------------
// Test sequence
// ------------------------------------------------------------
// Phase 1: staircase, forward.   duty = +0.1 .. +1.0
// Phase 2: staircase, reverse.   duty = -0.1 .. -1.0
// Phase 3: repeated steps 0 -> STEP_DUTY -> 0 for the dead time / tau_m
//
// The staircase gives the DC gain and the dead zone; the steps give the
// dead time and the motor time constant.
const uint16_t SETTLE_MS = 1000;   // zero-duty rest before/between phases
const uint16_t LEVEL_MS = 1500;    // hold time per staircase level
const uint8_t  N_LEVELS = 10;      // 0.1 .. 1.0

const float STEP_DUTY = 0.6f;
const uint16_t STEP_ON_MS = 1000;
const uint16_t STEP_OFF_MS = 1000;
const uint8_t  N_STEPS = 3;

unsigned long sequenceStartMs = 0;
bool sequenceDone = false;

// ============================================================
// Encoders (identical decoding and scaling to the balance sketch)
// ============================================================
class WheelAngleSamplerA {
private:
  int pin_a, pin_b, pin_a_prev, pin_b_prev;
  long cnt_encoder;

public:
  WheelAngleSamplerA() {
    cnt_encoder = 0;
    pinMode(ROTARY_ENCODER_A_PIN_1, INPUT_PULLUP);
    pinMode(ROTARY_ENCODER_A_PIN_2, INPUT_PULLUP);
    pin_a = digitalRead(ROTARY_ENCODER_A_PIN_1);
    pin_b = digitalRead(ROTARY_ENCODER_A_PIN_2);
    pin_a_prev = pin_a;
    pin_b_prev = pin_b;
  }

  float sampleInRad() {
    pin_a = digitalRead(ROTARY_ENCODER_A_PIN_1);
    pin_b = digitalRead(ROTARY_ENCODER_A_PIN_2);

    if (pin_a != pin_a_prev) {
      if (pin_a == pin_b) {
        cnt_encoder += 1;
      } else {
        cnt_encoder -= 1;
      }
    }

    if (pin_b != pin_b_prev) {
      if (pin_a == pin_b) {
        cnt_encoder -= 1;
      } else {
        cnt_encoder += 1;
      }
    }

    pin_a_prev = pin_a;
    pin_b_prev = pin_b;

    return float(cnt_encoder)
           * (20.0f / 14.0f)
           * (360.0f / 48.0f)
           / 360.0f
           * 2.0f * PI;
  }
};

class WheelAngleSamplerB {
private:
  int pin_a, pin_b, pin_a_prev, pin_b_prev;
  long cnt_encoder;

public:
  WheelAngleSamplerB() {
    cnt_encoder = 0;
    pinMode(ROTARY_ENCODER_B_PIN_1, INPUT_PULLUP);
    pinMode(ROTARY_ENCODER_B_PIN_2, INPUT_PULLUP);
    pin_a = digitalRead(ROTARY_ENCODER_B_PIN_1);
    pin_b = digitalRead(ROTARY_ENCODER_B_PIN_2);
    pin_a_prev = pin_a;
    pin_b_prev = pin_b;
  }

  float sampleInRad() {
    pin_a = digitalRead(ROTARY_ENCODER_B_PIN_1);
    pin_b = digitalRead(ROTARY_ENCODER_B_PIN_2);

    if (pin_a != pin_a_prev) {
      if (pin_a == pin_b) {
        cnt_encoder -= 1;
      } else {
        cnt_encoder += 1;
      }
    }

    if (pin_b != pin_b_prev) {
      if (pin_a == pin_b) {
        cnt_encoder += 1;
      } else {
        cnt_encoder -= 1;
      }
    }

    pin_a_prev = pin_a;
    pin_b_prev = pin_b;

    return float(cnt_encoder)
           * (20.0f / 12.0f)
           * (360.0f / 48.0f)
           / 360.0f
           * 2.0f * PI;
  }
};

WheelAngleSamplerA wheelAngleSamplerA;
WheelAngleSamplerB wheelAngleSamplerB;

float angleA = 0.0f;
float angleB = 0.0f;

// ============================================================
// Motor output (identical to the balance sketch)
// ============================================================
void setLeftDutyCycle(float pwm_value) {
  pwm_value = constrain(pwm_value, -1.0f, 1.0f);

  if (sign(pwm_value) == 0) {
    analogWrite(MOTOR_A_PIN_IN1, 255);
    analogWrite(MOTOR_A_PIN_IN2, 255);
    return;
  }

  if (sign(pwm_value) > 0) {
    analogWrite(MOTOR_A_PIN_IN1, 0);
    analogWrite(MOTOR_A_PIN_IN2, int(abs(pwm_value) * 255.0f));
  } else {
    analogWrite(MOTOR_A_PIN_IN2, 0);
    analogWrite(MOTOR_A_PIN_IN1, int(abs(pwm_value) * 255.0f));
  }
}

void setRightDutyCycle(float pwm_value) {
  pwm_value = constrain(pwm_value, -1.0f, 1.0f);

  if (sign(pwm_value) == 0) {
    analogWrite(MOTOR_B_PIN_IN1, 255);
    analogWrite(MOTOR_B_PIN_IN2, 255);
    return;
  }

  if (sign(pwm_value) > 0) {
    analogWrite(MOTOR_B_PIN_IN1, 0);
    analogWrite(MOTOR_B_PIN_IN2, int(abs(pwm_value) * 255.0f));
  } else {
    analogWrite(MOTOR_B_PIN_IN2, 0);
    analogWrite(MOTOR_B_PIN_IN1, int(abs(pwm_value) * 255.0f));
  }
}

void stopMotors() {
  setLeftDutyCycle(0.0f);
  setRightDutyCycle(0.0f);
}

// ============================================================
// Test sequence: elapsed time -> commanded duty
// ============================================================
float dutyForElapsed(unsigned long ms) {
  unsigned long t = ms;

  // ---- rest ----
  if (t < SETTLE_MS) {
    return 0.0f;
  }
  t -= SETTLE_MS;

  // ---- phase 1: forward staircase ----
  const unsigned long stairMs = (unsigned long)LEVEL_MS * N_LEVELS;

  if (t < stairMs) {
    uint8_t level = t / LEVEL_MS;
    return 0.1f * float(level + 1);
  }
  t -= stairMs;

  if (t < SETTLE_MS) {
    return 0.0f;
  }
  t -= SETTLE_MS;

  // ---- phase 2: reverse staircase ----
  if (t < stairMs) {
    uint8_t level = t / LEVEL_MS;
    return -0.1f * float(level + 1);
  }
  t -= stairMs;

  if (t < SETTLE_MS) {
    return 0.0f;
  }
  t -= SETTLE_MS;

  // ---- phase 3: repeated steps ----
  const unsigned long stepPeriod = (unsigned long)STEP_ON_MS + STEP_OFF_MS;
  const unsigned long stepsMs = stepPeriod * N_STEPS;

  if (t < stepsMs) {
    return (t % stepPeriod) < STEP_ON_MS ? STEP_DUTY : 0.0f;
  }

  // ---- done ----
  return NAN;
}

// ============================================================
void setup() {
  Serial.begin(115200);

  pinMode(MOTOR_A_PIN_IN1, OUTPUT);
  pinMode(MOTOR_A_PIN_IN2, OUTPUT);
  pinMode(MOTOR_B_PIN_IN1, OUTPUT);
  pinMode(MOTOR_B_PIN_IN2, OUTPUT);

  stopMotors();

  Serial.println("# measure_km");
  Serial.println("# columns: t_ms,duty,phiA,phiB");
  Serial.println("# phiA/phiB are wheel angles in rad (gear ratio included)");
  Serial.println("# put the robot on a stand: both wheels must spin freely");
  Serial.println("# no MsTimer2: every timer is left free so D3/D11 can PWM");
  Serial.println("# starting in 3 s");

  delay(3000);

  nextControlUs = micros();
  sequenceStartMs = millis();
  timePrev = micros() * 1.0e-6f;

  Serial.println("# BEGIN");
}

void loop() {
  // ----------------------------------------------------------
  // Fast encoder polling (same cadence as the balance sketch)
  // ----------------------------------------------------------
  timeNow = micros() * 1.0e-6f;

  if ((timeNow - timePrev) > ENCODER_POLL_SEC) {
    angleA = wheelAngleSamplerA.sampleInRad();
    angleB = wheelAngleSamplerB.sampleInRad();
    timePrev = timeNow;
  }

  // ----------------------------------------------------------
  // 10 ms control cycle, scheduled without any hardware timer
  // ----------------------------------------------------------
  if ((long)(micros() - nextControlUs) < 0) {
    return;
  }
  nextControlUs += CONTROL_PERIOD_US;

  if (sequenceDone) {
    return;
  }

  const unsigned long elapsed = millis() - sequenceStartMs;
  const float duty = dutyForElapsed(elapsed);

  if (isnan(duty)) {
    stopMotors();
    sequenceDone = true;
    Serial.println("# END");
    return;
  }

  setLeftDutyCycle(duty);
  setRightDutyCycle(duty);

  Serial.print(elapsed);
  Serial.print(',');
  Serial.print(duty, 3);
  Serial.print(',');
  Serial.print(angleA, 4);
  Serial.print(',');
  Serial.println(angleB, 4);
}
