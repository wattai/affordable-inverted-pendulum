// ============================================================
// Self-balancing robot: R=1.4 / R=1.6 selectable 4-state LQR
//
// Base:
//   User's corrected 1-wheel + 6-stage input-delay observer code.
//
// Main policy of this version:
//   1) Preserve the experimentally best 4-state balance architecture.
//   2) Keep left/right synchronization kWheelSync = 1.20.
//   3) Remove the unsuccessful delta-u controller.
//   4) Compare only R=1.4 and R=1.6 by changing one enum line.
//   5) Keep the optional slow position-hold outer loop disabled by default.
//   6) Keep Serial logging at 20 Hz.
//
// IMPORTANT:
//   angleSpeedAverageA/B behavior is intentionally preserved from the
//   currently stable code. Do not "fix" this into conventional wheel-speed
//   feedback unless the lower loop is retuned from scratch.
//
// NOTE:
//   MsTimer2 is also intentionally preserved here so that this test changes
//   as little as possible relative to the stable baseline.
// ============================================================

#define sign(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : 0))

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <MsTimer2.h>

// ------------------------------------------------------------
// Global timing
// ------------------------------------------------------------
volatile unsigned long previousTime = 0;
volatile bool readSensorFlag = false;

const int CONTROL_PERIOD_MS = 10;
const float CONTROL_DT_SEC = 0.010f;

// ------------------------------------------------------------
// BNO055
// ------------------------------------------------------------
Adafruit_BNO055 bno = Adafruit_BNO055(55);

float bodyAngleOffset = 0.0f;
float bodyAngleInRad = 0.0f;
float bodyAngleSpeedInRadPerSec = 0.0f;

// ------------------------------------------------------------
// Motor pins
// ------------------------------------------------------------
const int MOTOR_A_PIN_IN1 = 3;
const int MOTOR_A_PIN_IN2 = 9;
const int MOTOR_B_PIN_IN1 = 10;
const int MOTOR_B_PIN_IN2 = 11;

// ------------------------------------------------------------
// Encoder pins
// ------------------------------------------------------------
const int ROTARY_ENCODER_A_PIN_1 = 4;
const int ROTARY_ENCODER_A_PIN_2 = 2;
const int ROTARY_ENCODER_B_PIN_1 = 8;
const int ROTARY_ENCODER_B_PIN_2 = 7;

// ------------------------------------------------------------
// Physical/model parameters
// ------------------------------------------------------------
float T_sample = 0.010f;

float r_wheel = 2.7f * 0.01f;
float g = 9.8f;
float m_body = 350.0f * 0.001f;
float h_body = 8.0f * 0.01f;
float m_wheel = 10.0f * 0.001f;

float T_pendulum = 0.75f;
float I_eff =
    pow(T_pendulum, 2) * m_body * g * h_body
    / (4.0f * pow(PI, 2));

float tau_motor = 0.07f;

// ------------------------------------------------------------
// Balance-controller selection
// ------------------------------------------------------------
// The real robot was much more stable with the ordinary 4-state LQR
// than with the augmented delta-u controller, so this version removes
// the delta-u controller completely.
//
// Only R=1.4 and R=1.6 are retained for a clean A/B comparison.
// Change ONE line below to switch between them.
//
//   MODE_R14_4STATE : current experimentally best setting
//   MODE_R16_4STATE : next slightly lower-energy test
//
enum BalanceControllerMode : uint8_t {
  MODE_R14_4STATE = 0,
  MODE_R16_4STATE = 1
};

// Start from the current best result.
// Change only this line to MODE_R16_4STATE for the next experiment.
const BalanceControllerMode BALANCE_CONTROLLER_MODE = MODE_R14_4STATE;

// ------------------------------------------------------------
// R = 1.4 discrete LQR
// ------------------------------------------------------------
// Design basis:
//   Q = diag(10, 1, 1, 1, 0, 0, 0, 0, 0, 0)
//   R = 1.4
//
// Full gain calculated from the embedded A_d/B_d:
//   [-359.683463, -42.940531, -0.765166, -5.268757,
//      0.634256,   0.556236,  0.486689,  0.423728,
//      0.367017,   0.315662]
//
// As in the successful real-robot tests, only the first four gains
// are applied directly to the control command.
const float K_R14_THETA = -350.441115f;  // actually it assumes R=1.0
const float K_R14_THETA_DOT = -41.839831f;  // actually it assumes R=1.0
const float K_R14_WHEEL_ANGLE =  0.0f;  // zero to disable angle control  // actually it assumes R=1.0
const float K_R14_WHEEL_SPEED = -5.073251f;  // actually it assumes R=1.0
// const float K_R14_THETA = -335.154543f;
// const float K_R14_THETA_DOT = -40.013031f;
// // const float K_R14_WHEEL_ANGLE = -0.765166f;
// const float K_R14_WHEEL_ANGLE =  0.0f;  // zero to disable angle control
// const float K_R14_WHEEL_SPEED = -4.857388f;

// ------------------------------------------------------------
// R = 1.6 discrete LQR
// ------------------------------------------------------------
// Same Q and same plant model; only input penalty R is increased
// from 1.4 to 1.6.
//
// Full gain:
//   [-353.293860, -42.177194, -0.716797, -5.174422,
//      0.623040,   0.546368,  0.478140,  0.416533,
//      0.361206,   0.311237]
//
// Again, only the first four gains are used in the successful
// 4-state feedback architecture.
const float K_R16_THETA = -353.293860f;
const float K_R16_THETA_DOT = -42.177194f;
// const float K_R16_WHEEL_ANGLE = -0.716797f;
const float K_R16_WHEEL_ANGLE = 0.0f;  // zero to disable angle control
const float K_R16_WHEEL_SPEED = -5.174422f;

// Previous command is used ONLY for diagnostics (dU logging).
// It does not affect the controller.
float previousCommandForLog = 0.0f;

// Original delay-state gain names kept for compatibility/debug reference.
// The z states remain active inside the observer/model, but are not directly
// fed back into u_WheelAngleSpeed.
float gainZ1 = 0.667347f;
float gainZ2 = 0.585332f;
float gainZ3 = 0.511800f;
float gainZ4 = 0.445059f;
float gainZ5 = 0.384120f;
float gainZ6 = 0.328582f;

// ------------------------------------------------------------
// Left/right synchronization
// ------------------------------------------------------------
// Only this gain is intentionally kept stronger than the old baseline.
float kWheelSync = 1.20f;

// ------------------------------------------------------------
// Optional slow position-hold outer loop
// ------------------------------------------------------------
// FIRST TEST:
//   Leave this false. This gives the cleanest comparison against the
//   previously stable corrected code.
//
// AFTER confirming the fore/aft "gikogiko" is back to the previous level:
//   Set true and tune K_POSITION_TO_LEAN very slowly.
const bool ENABLE_SLOW_POSITION_HOLD = false;

// Update the slow outer loop at 10 Hz while the balance loop remains 100 Hz.
const uint8_t POSITION_LOOP_DIVIDER = 10;

// [rad body reference] / [rad wheel-position error]
float K_POSITION_TO_LEAN = 0.0010f;

// Limit the slow tilt correction to +/-0.5 deg initially.
const float MAX_POSITION_TRIM_RAD = 0.5f * PI / 180.0f;

// Low-pass filtering of the slow position trim.
const float POSITION_TRIM_ALPHA = 0.90f;

float wheelPositionReference = 0.0f;
float bodyAngleTrimFromPosition = 0.0f;
uint8_t positionLoopCounter = 0;

// ------------------------------------------------------------
// Delay-state estimates
// ------------------------------------------------------------
float z1Estimated = 0.0f;
float z2Estimated = 0.0f;
float z3Estimated = 0.0f;
float z4Estimated = 0.0f;
float z5Estimated = 0.0f;
float z6Estimated = 0.0f;

float z1EstimatedPrev = 0.0f;
float z2EstimatedPrev = 0.0f;
float z3EstimatedPrev = 0.0f;
float z4EstimatedPrev = 0.0f;
float z5EstimatedPrev = 0.0f;
float z6EstimatedPrev = 0.0f;

// ------------------------------------------------------------
// Encoder sampling variables
// ------------------------------------------------------------
float angleA = 0.0f;
float angleB = 0.0f;

float anglePrevA = 0.0f;
float anglePrevB = 0.0f;

float angleSpeedA = 0.0f;
float angleSpeedB = 0.0f;

float angleSpeedAverageA = 0.0f;
float angleSpeedAverageB = 0.0f;

float timeNow1 = 0.0f;
float timePrev1 = 0.0f;

const float angleSensorTimeDeltaSec = 0.0001f;
const float alpha = 0.8f;

// ------------------------------------------------------------
// Observer state
// ------------------------------------------------------------
float bodyAngleInRadEstimated = 0.0f;
float bodyAngleSpeedInRadPerSecEstimated = 0.0f;
float wheelAngleInRadEstimated = 0.0f;
float wheelAngleSpeedInRadPerSecEstimated = 0.0f;

float bodyAngleInRadEstimatedPrev = 0.0f;
float bodyAngleSpeedInRadPerSecEstimatedPrev = 0.0f;
float wheelAngleInRadEstimatedPrev = 0.0f;
float wheelAngleSpeedInRadPerSecEstimatedPrev = 0.0f;

float wheelAngleInRad = 0.0f;
float wheelAngleSpeedInRadPerSec = 0.0f;

float u_WheelAngleSpeed = 0.0f;

// ------------------------------------------------------------
// Lower motor loop
// ------------------------------------------------------------
// Preserve the asymmetry that was present in the stable code.
float kpA = 0.025f;
float kpB = 0.030f;
float kiA = 0.040f;
float kiB = 0.050f;

float eAI = 0.0f;
float eBI = 0.0f;

// ------------------------------------------------------------
// Diagnostics / safety
// ------------------------------------------------------------
const uint8_t LOG_DIVIDER = 5;   // 100 Hz / 5 = 20 Hz
uint8_t logCounter = 0;

const float FALL_ANGLE_LIMIT_RAD = 35.0f * PI / 180.0f;

// ============================================================
// Body angle sampler
// ============================================================
class BodyAngleSampler {
private:
  float current_value;
  float prev_value;

public:
  BodyAngleSampler() {
    current_value = 0.0f;
    prev_value = 0.0f;
  }

  float sampleInRad() {
    sensors_event_t orientation_event;
    bno.getEvent(&orientation_event, Adafruit_BNO055::VECTOR_EULER);

    prev_value = current_value;

    // Preserve the same behavior around the problematic Euler region.
    if (abs(orientation_event.orientation.z) < 20.0f) {
      return (prev_value - bodyAngleOffset) * PI / 180.0f;
    }

    current_value = orientation_event.orientation.z;
    return (current_value - bodyAngleOffset) * PI / 180.0f;
  }
};

// ============================================================
// Encoder A
// ============================================================
class WheelAngleSamplerA {
private:
  int pin_a;
  int pin_b;
  int pin_a_prev;
  int pin_b_prev;
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

// ============================================================
// Encoder B
// ============================================================
class WheelAngleSamplerB {
private:
  int pin_a;
  int pin_b;
  int pin_a_prev;
  int pin_b_prev;
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

// ============================================================
// BNO gyro
// ============================================================
float getBodyAngleSpeedInRadPerSec() {
  sensors_event_t gyro_event;
  bno.getEvent(&gyro_event, Adafruit_BNO055::VECTOR_GYROSCOPE);
  return -gyro_event.gyro.x;
}

// ============================================================
// Motor output
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
// Objects
// ============================================================
BodyAngleSampler bodyAngleSampler;
WheelAngleSamplerA wheelAngleSamplerA;
WheelAngleSamplerB wheelAngleSamplerB;

// ============================================================
// Timer callback
// ============================================================
void runController() {
  readSensorFlag = true;
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);

  if (!bno.begin()) {
    Serial.println("BNO055 initialization failed.");

    while (1) {
      stopMotors();
    }
  }

  bno.setExtCrystalUse(true);
  bno.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);

  pinMode(MOTOR_A_PIN_IN1, OUTPUT);
  pinMode(MOTOR_A_PIN_IN2, OUTPUT);
  pinMode(MOTOR_B_PIN_IN1, OUTPUT);
  pinMode(MOTOR_B_PIN_IN2, OUTPUT);

  pinMode(ROTARY_ENCODER_A_PIN_1, INPUT_PULLUP);
  pinMode(ROTARY_ENCODER_A_PIN_2, INPUT_PULLUP);
  pinMode(ROTARY_ENCODER_B_PIN_1, INPUT_PULLUP);
  pinMode(ROTARY_ENCODER_B_PIN_2, INPUT_PULLUP);

  stopMotors();

  // ----------------------------------------------------------
  // Body angle offset calibration
  // ----------------------------------------------------------
  int angleCalibrationCnt = 0;

  for (int i = 0; i < 300; i++) {
    sensors_event_t orientation_event;
    bno.getEvent(&orientation_event, Adafruit_BNO055::VECTOR_EULER);

    if (abs(orientation_event.orientation.z) > 20.0f) {
      bodyAngleOffset += orientation_event.orientation.z;
      angleCalibrationCnt++;
    }

    delay(10);
  }

  if (angleCalibrationCnt > 0) {
    bodyAngleOffset /= float(angleCalibrationCnt);
  } else {
    bodyAngleOffset = 0.0f;
  }

  // Encoders start from zero in these sampler classes.
  wheelPositionReference = 0.0f;

  previousTime = micros();

  // Preserve stable timing architecture for this comparison.
  MsTimer2::set(CONTROL_PERIOD_MS, runController);
  MsTimer2::start();
}

// ============================================================
// Main loop
// ============================================================
void loop() {
  // ----------------------------------------------------------
  // Fast encoder polling
  // ----------------------------------------------------------
  timeNow1 = micros() * 0.001f * 0.001f;

  if ((timeNow1 - timePrev1) > angleSensorTimeDeltaSec) {
    angleA = wheelAngleSamplerA.sampleInRad();
    angleB = wheelAngleSamplerB.sampleInRad();

    float dt = timeNow1 - timePrev1;

    if (dt > 0.0f) {
      angleSpeedA =
          alpha * angleSpeedA
          + (1.0f - alpha) * (angleA - anglePrevA) / dt;

      angleSpeedB =
          alpha * angleSpeedB
          + (1.0f - alpha) * (angleB - anglePrevB) / dt;
    }

    // --------------------------------------------------------
    // IMPORTANT:
    // Preserve the effective behavior of the current stable code.
    // --------------------------------------------------------
    angleSpeedAverageA +=
        angleSpeedAverageA
        / (CONTROL_DT_SEC / angleSensorTimeDeltaSec);

    angleSpeedAverageB +=
        angleSpeedAverageB
        / (CONTROL_DT_SEC / angleSensorTimeDeltaSec);

    anglePrevA = angleA;
    anglePrevB = angleB;
    timePrev1 = timeNow1;
  }

  if (!readSensorFlag) {
    return;
  }

  // ----------------------------------------------------------
  // Control-cycle timing
  // ----------------------------------------------------------
  unsigned long currentTime = micros();
  unsigned long elapsedTime = currentTime - previousTime;
  previousTime = currentTime;

  // ----------------------------------------------------------
  // Sensors
  // ----------------------------------------------------------
  bodyAngleInRad = bodyAngleSampler.sampleInRad();
  bodyAngleSpeedInRadPerSec = getBodyAngleSpeedInRadPerSec();

  wheelAngleInRad = (angleA + angleB) * 0.5f;
  wheelAngleSpeedInRadPerSec =
      (angleSpeedAverageA + angleSpeedAverageB) * 0.5f;

  // ----------------------------------------------------------
  // Safety cutoff
  // ----------------------------------------------------------
  if (abs(bodyAngleInRad) > FALL_ANGLE_LIMIT_RAD) {
    stopMotors();

    eAI = 0.0f;
    eBI = 0.0f;
    previousCommandForLog = 0.0f;
    bodyAngleTrimFromPosition = 0.0f;

    readSensorFlag = false;
    angleSpeedAverageA = 0.0f;
    angleSpeedAverageB = 0.0f;

    return;
  }

  // ==========================================================
  // Observer prediction
  // ==========================================================
  bodyAngleInRadEstimated = 0.0f;
  bodyAngleSpeedInRadPerSecEstimated = 0.0f;
  wheelAngleInRadEstimated = 0.0f;
  wheelAngleSpeedInRadPerSecEstimated = 0.0f;

  z1Estimated = 0.0f;
  z2Estimated = 0.0f;
  z3Estimated = 0.0f;
  z4Estimated = 0.0f;
  z5Estimated = 0.0f;
  z6Estimated = 0.0f;

  bodyAngleInRadEstimated +=
      1.003511f * bodyAngleInRadEstimatedPrev
      + 0.010012f * bodyAngleSpeedInRadPerSecEstimatedPrev
      + 0.000132f * wheelAngleSpeedInRadPerSecEstimatedPrev
      - 0.000109f * z1EstimatedPrev
      - 0.000020f * z2EstimatedPrev
      - 0.000003f * z3EstimatedPrev;

  bodyAngleSpeedInRadPerSecEstimated +=
      0.702660f * bodyAngleInRadEstimatedPrev
      + 1.003511f * bodyAngleSpeedInRadPerSecEstimatedPrev
      + 0.025772f * wheelAngleSpeedInRadPerSecEstimatedPrev
      - 0.019247f * z1EstimatedPrev
      - 0.005332f * z2EstimatedPrev
      - 0.001024f * z3EstimatedPrev
      - 0.000150f * z4EstimatedPrev
      - 0.000018f * z5EstimatedPrev
      - 0.000002f * z6EstimatedPrev;

  wheelAngleInRadEstimated +=
      1.000000f * wheelAngleInRadEstimatedPrev
      + 0.009319f * wheelAngleSpeedInRadPerSecEstimatedPrev
      + 0.000562f * z1EstimatedPrev
      + 0.000103f * z2EstimatedPrev
      + 0.000015f * z3EstimatedPrev
      + 0.000002f * z4EstimatedPrev;

  wheelAngleSpeedInRadPerSecEstimated +=
      0.866878f * wheelAngleSpeedInRadPerSecEstimatedPrev
      + 0.099396f * z1EstimatedPrev
      + 0.027555f * z2EstimatedPrev
      + 0.005295f * z3EstimatedPrev
      + 0.000775f * z4EstimatedPrev
      + 0.000092f * z5EstimatedPrev
      + 0.000009f * z6EstimatedPrev;

  // Corrected z1 transition
  z1Estimated +=
      0.548812f * z1EstimatedPrev
      + 0.329287f * z2EstimatedPrev
      + 0.098786f * z3EstimatedPrev
      + 0.019757f * z4EstimatedPrev
      + 0.002964f * z5EstimatedPrev
      + 0.000356f * z6EstimatedPrev;

  z2Estimated +=
      0.548812f * z2EstimatedPrev
      + 0.329287f * z3EstimatedPrev
      + 0.098786f * z4EstimatedPrev
      + 0.019757f * z5EstimatedPrev
      + 0.002964f * z6EstimatedPrev;

  z3Estimated +=
      0.548812f * z3EstimatedPrev
      + 0.329287f * z4EstimatedPrev
      + 0.098786f * z5EstimatedPrev
      + 0.019757f * z6EstimatedPrev;

  z4Estimated +=
      0.548812f * z4EstimatedPrev
      + 0.329287f * z5EstimatedPrev
      + 0.098786f * z6EstimatedPrev;

  z5Estimated +=
      0.548812f * z5EstimatedPrev
      + 0.329287f * z6EstimatedPrev;

  z6Estimated +=
      0.548812f * z6EstimatedPrev;

  // Input contribution
  wheelAngleSpeedInRadPerSecEstimated +=
      0.000001f * u_WheelAngleSpeed;

  z1Estimated += 0.000039f * u_WheelAngleSpeed;
  z2Estimated += 0.000390f * u_WheelAngleSpeed;
  z3Estimated += 0.003358f * u_WheelAngleSpeed;
  z4Estimated += 0.023115f * u_WheelAngleSpeed;
  z5Estimated += 0.121901f * u_WheelAngleSpeed;
  z6Estimated += 0.451188f * u_WheelAngleSpeed;

  // ==========================================================
  // Observer correction
  // ==========================================================
  float e1 =
      bodyAngleInRad
      - bodyAngleInRadEstimatedPrev;

  float e2 =
      bodyAngleSpeedInRadPerSec
      - bodyAngleSpeedInRadPerSecEstimatedPrev;

  float e3 =
      wheelAngleInRad
      - wheelAngleInRadEstimatedPrev;

  float e4 =
      wheelAngleSpeedInRadPerSec
      - wheelAngleSpeedInRadPerSecEstimatedPrev;

  bodyAngleInRadEstimated +=
      0.604600f * e1
      + 0.064392f * e2
      + 0.000002f * e3
      + 0.000048f * e4;

  bodyAngleSpeedInRadPerSecEstimated +=
      0.064392f * e1
      + 0.662611f * e2
      + 0.000010f * e3
      + 0.000278f * e4;

  wheelAngleInRadEstimated +=
      -0.000002f * e1
      + 0.000010f * e2
      + 0.618044f * e3
      + 0.000941f * e4;

  wheelAngleSpeedInRadPerSecEstimated +=
      0.000048f * e1
      + 0.000278f * e2
      + 0.000941f * e3
      + 0.602982f * e4;

  z1Estimated +=
      0.005301f * e1
      - 0.020367f * e2
      + 0.000800f * e3
      + 0.137743f * e4;

  z2Estimated +=
      0.003264f * e1
      - 0.012777f * e2
      + 0.000452f * e3
      + 0.085731f * e4;

  z3Estimated +=
      0.001733f * e1
      - 0.006923f * e2
      + 0.000230f * e3
      + 0.046081f * e4;

  z4Estimated +=
      0.000760f * e1
      - 0.003108f * e2
      + 0.000096f * e3
      + 0.020510f * e4;

  z5Estimated +=
      0.000247f * e1
      - 0.001037f * e2
      + 0.000030f * e3
      + 0.006781f * e4;

  z6Estimated +=
      0.000045f * e1
      - 0.000195f * e2
      + 0.000005f * e3
      + 0.001260f * e4;

  wheelAngleSpeedInRadPerSecEstimated =
      constrain(
          wheelAngleSpeedInRadPerSecEstimated,
          -20.0f * 2.0f * PI,
           20.0f * 2.0f * PI);

  // Save observer states
  bodyAngleInRadEstimatedPrev =
      bodyAngleInRadEstimated;

  bodyAngleSpeedInRadPerSecEstimatedPrev =
      bodyAngleSpeedInRadPerSecEstimated;

  wheelAngleInRadEstimatedPrev =
      wheelAngleInRadEstimated;

  wheelAngleSpeedInRadPerSecEstimatedPrev =
      wheelAngleSpeedInRadPerSecEstimated;

  z1EstimatedPrev = z1Estimated;
  z2EstimatedPrev = z2Estimated;
  z3EstimatedPrev = z3Estimated;
  z4EstimatedPrev = z4Estimated;
  z5EstimatedPrev = z5Estimated;
  z6EstimatedPrev = z6Estimated;

  // ==========================================================
  // Optional slow position-hold outer loop
  // ==========================================================
  if (ENABLE_SLOW_POSITION_HOLD) {
    positionLoopCounter++;

    if (positionLoopCounter >= POSITION_LOOP_DIVIDER) {
      positionLoopCounter = 0;

      float positionError =
          wheelAngleInRad - wheelPositionReference;

      float targetTrim =
          -K_POSITION_TO_LEAN * positionError;

      targetTrim =
          constrain(
              targetTrim,
              -MAX_POSITION_TRIM_RAD,
               MAX_POSITION_TRIM_RAD);

      bodyAngleTrimFromPosition =
          POSITION_TRIM_ALPHA * bodyAngleTrimFromPosition
          + (1.0f - POSITION_TRIM_ALPHA) * targetTrim;
    }
  } else {
    bodyAngleTrimFromPosition = 0.0f;
  }

  // ==========================================================
  // Balance controller
  // ==========================================================
  // The optional slow position loop changes only the body-angle reference.
  float bodyAngleForControl =
      bodyAngleInRadEstimated
      - bodyAngleTrimFromPosition;

  float newCommand = 0.0f;

  if (BALANCE_CONTROLLER_MODE == MODE_R14_4STATE) {
    // --------------------------------------------------------
    // R = 1.4 : current experimentally best setting
    // --------------------------------------------------------
    newCommand =
        -K_R14_THETA * bodyAngleForControl
        -K_R14_THETA_DOT * bodyAngleSpeedInRadPerSecEstimated
        -K_R14_WHEEL_ANGLE * wheelAngleInRadEstimated
        -K_R14_WHEEL_SPEED * wheelAngleSpeedInRadPerSecEstimated;

  } else {
    // --------------------------------------------------------
    // R = 1.6 : slightly stronger input-energy penalty
    // --------------------------------------------------------
    newCommand =
        -K_R16_THETA * bodyAngleForControl
        -K_R16_THETA_DOT * bodyAngleSpeedInRadPerSecEstimated
        -K_R16_WHEEL_ANGLE * wheelAngleInRadEstimated
        -K_R16_WHEEL_SPEED * wheelAngleSpeedInRadPerSecEstimated;
  }

  // Diagnostic only: how much the requested common command changed
  // from the preceding control cycle.
  float deltaUForLog =
      newCommand - previousCommandForLog;

  u_WheelAngleSpeed = newCommand;
  previousCommandForLog = newCommand;

  // ==========================================================
  // Left/right synchronization
  // ==========================================================
  float wheelDiff = angleB - angleA;

  float eA =
      u_WheelAngleSpeed
      - angleSpeedAverageA
      + kWheelSync * wheelDiff;

  float eB =
      u_WheelAngleSpeed
      - angleSpeedAverageB
      - kWheelSync * wheelDiff;

  // Preserve the leaky PI behavior
  eAI =
      eA * CONTROL_DT_SEC
      + eAI * 0.99f;

  eBI =
      eB * CONTROL_DT_SEC
      + eBI * 0.99f;

  float uA =
      kpA * eA
      + kiA * eAI;

  float uB =
      kpB * eB
      + kiB * eBI;

  uA = constrain(uA, -1.0f, 1.0f);
  uB = constrain(uB, -1.0f, 1.0f);

  setLeftDutyCycle(uA);
  setRightDutyCycle(uB);

  // ==========================================================
  // 20 Hz diagnostics
  // ==========================================================
  logCounter++;

  if (logCounter >= LOG_DIVIDER) {
    logCounter = 0;

    Serial.print("dt_us:");
    Serial.print(elapsedTime);

    Serial.print(",angle:");
    Serial.print(bodyAngleInRad, 6);

    Serial.print(",gyro:");
    Serial.print(bodyAngleSpeedInRadPerSec, 6);

    Serial.print(",wheelPos:");
    Serial.print(wheelAngleInRad, 6);

    Serial.print(",wheelPosEst:");
    Serial.print(wheelAngleInRadEstimated, 6);

    Serial.print(",wheelSpeedEst:");
    Serial.print(wheelAngleSpeedInRadPerSecEstimated, 6);

    Serial.print(",wheelDiff:");
    Serial.print(wheelDiff, 6);

    Serial.print(",posTrim:");
    Serial.print(bodyAngleTrimFromPosition, 6);

    Serial.print(",mode:");
    Serial.print((int)BALANCE_CONTROLLER_MODE);

    Serial.print(",uPrev:");
    Serial.print(previousCommandForLog, 6);

    Serial.print(",dU:");
    Serial.print(deltaUForLog, 6);

    Serial.print(",uRef:");
    Serial.print(u_WheelAngleSpeed, 6);

    Serial.print(",pwmA:");
    Serial.print(uA, 6);

    Serial.print(",pwmB:");
    Serial.print(uB, 6);

    Serial.println();
  }

  // ----------------------------------------------------------
  // End of control cycle
  // ----------------------------------------------------------
  readSensorFlag = false;

  angleSpeedAverageA = 0.0f;
  angleSpeedAverageB = 0.0f;
}
