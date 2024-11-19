// 2 wheels model with considering of the input delay.
#define sign(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : 0))

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <MsTimer2.h>
#include <avr/pgmspace.h>  // Needed for PROGMEM

volatile unsigned long previousTime = 0;  // 前回のタイムスタンプ

// BNO055センサのインスタンス
Adafruit_BNO055 bno = Adafruit_BNO055(55);
volatile bool readSensorFlag = false;  // 割り込みでセンサーを読むためのフラグ

int MOTOR_A_PIN_IN1 = 3;
int MOTOR_A_PIN_IN2 = 9;
int MOTOR_B_PIN_IN1 = 10;
int MOTOR_B_PIN_IN2 = 11;

int ROTARY_ENCODER_A_PIN_1 = 4;
int ROTARY_ENCODER_A_PIN_2 = 2;
int ROTARY_ENCODER_B_PIN_1 = 8;
int ROTARY_ENCODER_B_PIN_2 = 7;

float bodyAngleInRad = 0;
float bodyAngleSpeedInRadPerSec = 0;
float wheelAngleSpeedInRadPerSec = 0;

// set parameters
float timeNow1 = 0;
float timePrev1 = 0;
float timeNow2 = 0;
float timePrev2 = 0;

int timeDeltaMilli = 10;  // CONTROL PERIOD.
float controlTimeDeltaSec = (float)timeDeltaMilli * 0.001;
float angleSensorTimeDeltaSec = 0.0001;

float bodyAngleOffset = 0;

float r_wheel = 2.7 * 0.01;  // wheel radius [m]
float g = 9.8;               // gravity acceleration [m/s^2]
float m_body = 250 * 0.001;  // body mass [kg]
float h_body = 10.5 * 0.01;   // body height between wheel axis and body mass center [m]

float m_wheel = 10.0 * 0.001;  // [kg]

float T_pendulum = 0.75;  // 振り子周期 [s]
float I_eff = pow(T_pendulum, 2) * m_body * g * h_body / (4 * pow(PI, 2));  // effective moment of inertia [?]

float tau_motor = 0.07;  // motor time consistency [s]

const unsigned int NUM_STATE_VARIABLES = 6;
const unsigned int NUM_OBSERVABLE_VARIABLES = 6;
const unsigned int NUM_CONTROL_INPUTS = 2;


float stateVariablesEstimated[NUM_STATE_VARIABLES] = {
  0,
  0,
  0,
  0,
  0,
  0,
};
float stateVariablesPrevious[NUM_STATE_VARIABLES] = {
  0,
  0,
  0,
  0,
  0,
  0,
};
const float matrixA[NUM_STATE_VARIABLES][NUM_STATE_VARIABLES] = {
 {1.003511, 0.010012, 0.000000, 0.000066, 0.000000, 0.000066},
 {0.702660, 1.003511, 0.000000, 0.012886, 0.000000, 0.012886},
 {0.000000, 0.000000, 1.000000, 0.009319, 0.000000, 0.000000},
 {0.000000, 0.000000, 0.000000, 0.866878, 0.000000, 0.000000},
 {0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.009319},
 {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.866878}
};
const float matrixB[NUM_STATE_VARIABLES][2] = {
 {-0.000066, -0.000066},
 {-0.012886, -0.012886},
 {0.000681, 0.000000},
 {0.133122, 0.000000},
 {0.000000, 0.000681},
 {0.000000, 0.133122}
};
const float observerGainMatrix[NUM_STATE_VARIABLES][NUM_OBSERVABLE_VARIABLES] = {
 {0.618205, 0.023604, -0.000001, 0.000050, -0.000001, 0.000050},
 {0.447182, 0.931141, 0.000006, 0.011848, 0.000006, 0.011848},
 {-0.000001, 0.000004, 0.618050, 0.008770, -0.000000, -0.000000},
 {-0.000010, 0.000063, 0.000215, 0.792704, -0.000000, -0.000000},
 {-0.000001, 0.000004, -0.000000, -0.000000, 0.618050, 0.008770},
 {-0.000010, 0.000063, -0.000000, -0.000000, 0.000215, 0.792704}
};

const float feedbackGainsLeft[NUM_STATE_VARIABLES] = {
  -286.153309, -34.160850, 0.034129, -0.979034, -0.813756, -2.813608,
};
const float feedbackGainsRight[NUM_STATE_VARIABLES] = {
  -286.153309, -34.160850, -0.813756, -2.813608, 0.034129, -0.979034
};

float kpA = 0.07;
float kpB = 0.07;
float kiA = 0.03;
float kiB = 0.03;
float eAI = 0.0;
float eBI = 0.0;
float alpha = 0.8;

float leftWheelAngleInRad = 0;
float leftWheelAngleSpeedInRadPerSec = 0;
float rightWheelAngleInRad = 0;
float rightWheelAngleSpeedInRadPerSec = 0;

float u_LeftWheelAngleSpeed = 0;
float u_RightWheelAngleSpeed = 0;
float u_left_pwm_value = 0;
float u_right_pwm_value = 0;

float angleSpeedAverageA = 0;
float angleSpeedAverageB = 0;



void copyArray(float* source, float* destination, int size) {
    for (int i = 0; i < size; i++) {
        destination[i] = source[i];
    }
}

void initializeToZero(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = 0;
    }
}

void matrixVectorProduct(float* matrix, float* vec, float* result, int rows, int cols) {
    // 行列とベクトルの内積を計算（手動でインデックス計算）
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vec[j];
            // result[i] += pgm_read_float(&(matrix[i * cols + j])) * vec[j];
        }
    }
}

float dotProduct(float* vec1, float* vec2, int size) {
    float result = 0;

    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

class BodyAngleSampler {
private:
  float current_value;
  float prev_value;

public:
  BodyAngleSampler() {
    current_value = 0;
    prev_value = 0;
  }

  float sampleInRad() {
    // 姿勢角（オイラー角）を取得
    sensors_event_t orientation_event;
    bno.getEvent(&orientation_event, Adafruit_BNO055::VECTOR_EULER);
    prev_value = current_value;
    if (abs(orientation_event.orientation.z) < 20) {
      return (prev_value - bodyAngleOffset) * PI / 180;  // 22
    };
    current_value = orientation_event.orientation.z;
    return (current_value - bodyAngleOffset) * PI / 180;  // 22
  }
};


class WheelAngleSamplerA {
private:
  float current_value;
  float prev_value;
  int pin_a = digitalRead(ROTARY_ENCODER_A_PIN_1);
  int pin_b = digitalRead(ROTARY_ENCODER_A_PIN_2);
  int pin_a_prev = pin_a;
  int pin_b_prev = pin_b;
  int cnt_encoder;

public:
  WheelAngleSamplerA() {
    current_value = 0;
    prev_value = 0;
    cnt_encoder = 0;
    pinMode(ROTARY_ENCODER_A_PIN_1, INPUT_PULLUP);
    pinMode(ROTARY_ENCODER_A_PIN_2, INPUT_PULLUP);
  }

  float sampleInRad() {
    pin_a = digitalRead(ROTARY_ENCODER_A_PIN_1);
    pin_b = digitalRead(ROTARY_ENCODER_A_PIN_2);

    if (pin_a != pin_a_prev){
        if (pin_a == pin_b) {
          cnt_encoder += 1;
        } else {
          cnt_encoder -= 1;
        }
    }
    if (pin_b != pin_b_prev){
        if (pin_a == pin_b) {
          cnt_encoder -= 1;
        } else {
          cnt_encoder += 1;
        }
    }
    pin_a_prev = pin_a;
    pin_b_prev = pin_b;

    return float(cnt_encoder) * (20.0 / 14.0) * (360.0 / 48.0) * 1.0 / 360.0 * 2 * PI;
  }
};

class WheelAngleSamplerB {
private:
  float current_value;
  float prev_value;
  int pin_a = digitalRead(ROTARY_ENCODER_B_PIN_1);
  int pin_b = digitalRead(ROTARY_ENCODER_B_PIN_2);
  int pin_a_prev = pin_a;
  int pin_b_prev = pin_b;
  int cnt_encoder;

public:
  WheelAngleSamplerB() {
    current_value = 0;
    prev_value = 0;
    cnt_encoder = 0;
    pinMode(ROTARY_ENCODER_B_PIN_1, INPUT_PULLUP);
    pinMode(ROTARY_ENCODER_B_PIN_2, INPUT_PULLUP);
  }

  float sampleInRad() {
    pin_a = digitalRead(ROTARY_ENCODER_B_PIN_1);
    pin_b = digitalRead(ROTARY_ENCODER_B_PIN_2);

    if (pin_a != pin_a_prev){
        if (pin_a == pin_b) {
          cnt_encoder -= 1;
        } else {
          cnt_encoder += 1;
        }
    }
    if (pin_b != pin_b_prev){
        if (pin_a == pin_b) {
          cnt_encoder += 1;
        } else {
          cnt_encoder -= 1;
        }
    }
    pin_a_prev = pin_a;
    pin_b_prev = pin_b;

    return float(cnt_encoder) * (20.0 / 12.0) * (360.0 / 48.0) * 1.0 / 360.0 * 2 * PI;
  }
};


float getBodyAngleSpeedInRadPerSec() {
  // ジャイロスコープ（角速度）を取得
  sensors_event_t gyro_event;
  bno.getEvent(&gyro_event, Adafruit_BNO055::VECTOR_GYROSCOPE);
  return -gyro_event.gyro.x;
}

void setLeftDutyCycle(float pwm_value) {
  // pwm_value ranges from -1 to 1.
  if (sign(pwm_value) == 0) {
    analogWrite(MOTOR_A_PIN_IN1, 255);
    analogWrite(MOTOR_A_PIN_IN2, 255);
    return;
  }

  if (sign(pwm_value) == 1) {
    analogWrite(MOTOR_A_PIN_IN1, 0);
    analogWrite(MOTOR_A_PIN_IN2, int(abs(pwm_value) * 255));
  } else {
    analogWrite(MOTOR_A_PIN_IN2, 0);
    analogWrite(MOTOR_A_PIN_IN1, int(abs(pwm_value) * 255));
  }
}

void setRightDutyCycle(float pwm_value) {
  // pwm_value ranges from -1 to 1.
  if (sign(pwm_value) == 0) {
    analogWrite(MOTOR_B_PIN_IN1, 255);
    analogWrite(MOTOR_B_PIN_IN2, 255);
    return;
  }

  if (sign(pwm_value) == 1) {
    analogWrite(MOTOR_B_PIN_IN1, 0);
    analogWrite(MOTOR_B_PIN_IN2, int(abs(pwm_value) * 255));
  } else {
    analogWrite(MOTOR_B_PIN_IN2, 0);
    analogWrite(MOTOR_B_PIN_IN1, int(abs(pwm_value) * 255));
  }
}

BodyAngleSampler bodyAngleSampler;
WheelAngleSamplerA wheelAngleSamplerA;
WheelAngleSamplerB wheelAngleSamplerB;

void setup() {
  Serial.begin(115200);

  // BNO055センサの初期化
  if (!bno.begin()) {
    Serial.println("BNO055の初期化に失敗しました。");
    while (1);
  }

  // センサのキャリブレーション
  bno.setExtCrystalUse(true);
  bno.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);

  // put your setup code here, to run once:
  pinMode(MOTOR_A_PIN_IN1, OUTPUT);
  pinMode(MOTOR_A_PIN_IN2, OUTPUT);
  pinMode(MOTOR_B_PIN_IN1, OUTPUT);
  pinMode(MOTOR_B_PIN_IN2, OUTPUT);

  pinMode(ROTARY_ENCODER_A_PIN_1, INPUT_PULLUP);
  pinMode(ROTARY_ENCODER_A_PIN_2, INPUT_PULLUP);
  pinMode(ROTARY_ENCODER_B_PIN_1, INPUT_PULLUP);
  pinMode(ROTARY_ENCODER_B_PIN_2, INPUT_PULLUP);

  // calculate body angle offset.
  int angleCalibrationCnt = 0; 
  for (int i=0; i<300; i++) {
    sensors_event_t orientation_event;
    bno.getEvent(&orientation_event, Adafruit_BNO055::VECTOR_EULER);
    Serial.println(orientation_event.orientation.z);
    if (abs(orientation_event.orientation.z) > 20) {
      bodyAngleOffset += orientation_event.orientation.z;
      angleCalibrationCnt++;
    }
    delay(10);
  }
  bodyAngleOffset /= (float)angleCalibrationCnt;

  // MsTimer2の設定
  MsTimer2::set(timeDeltaMilli, runController);  // タイマー割り込みの間隔とコールバック関数を設定
  MsTimer2::start();  // タイマー割り込みを開始
}


void runController() {
  // ここではフラグだけをセットし、実際のセンサー読み取りはloop()で行う
  readSensorFlag = true;
}

float angleA = 0;
float angleB = 0;
float anglePrevA = 0;
float anglePrevB = 0;
float angleSpeedA = 0;
float angleSpeedB = 0;

void loop() {
  timeNow1 = micros() * 0.001 * 0.001;  // saved in [sec]

  if ((timeNow1 - timePrev1) > angleSensorTimeDeltaSec) {
    angleA = wheelAngleSamplerA.sampleInRad();
    angleB = wheelAngleSamplerB.sampleInRad();

    angleSpeedA = alpha * angleSpeedA + (1.0 - alpha) * (angleA - anglePrevA) / (timeNow1 - timePrev1);
    angleSpeedB = alpha * angleSpeedB + (1.0 - alpha) * (angleB - anglePrevB) / (timeNow1 - timePrev1);

    angleSpeedAverageA += angleSpeedAverageA / (controlTimeDeltaSec / angleSensorTimeDeltaSec);
    angleSpeedAverageB += angleSpeedAverageB / (controlTimeDeltaSec / angleSensorTimeDeltaSec);

    anglePrevA = angleA;
    anglePrevB = angleB;
    timePrev1 = timeNow1;
  }

  if (readSensorFlag) {
    // Time capture ------------------------------------
    // 割り込みは非同期で実行される
    unsigned long currentTime = micros();                    // 現在の時間を取得（マイクロ秒単位）
    unsigned long elapsedTime = currentTime - previousTime;  // 前回からの経過時間

    // 経過時間をシリアルモニタに出力
    Serial.print("Elapsed time: ");
    Serial.print(currentTime);
    Serial.print(" us");

    previousTime = currentTime;  // 現在の時間を前回の時間として保存
    // -------------------------------------------------

    // get sensor data
    bodyAngleInRad = bodyAngleSampler.sampleInRad();
    bodyAngleSpeedInRadPerSec = getBodyAngleSpeedInRadPerSec();
    leftWheelAngleInRad = angleA;
    leftWheelAngleSpeedInRadPerSec = angleSpeedAverageA;
    rightWheelAngleInRad = angleB;
    rightWheelAngleSpeedInRadPerSec = angleSpeedAverageB;

    // start controling
    initializeToZero(stateVariablesEstimated, NUM_STATE_VARIABLES);
    matrixVectorProduct((float*)matrixA, stateVariablesPrevious, stateVariablesEstimated, NUM_STATE_VARIABLES, NUM_STATE_VARIABLES);

    float us[NUM_CONTROL_INPUTS] = {
      u_LeftWheelAngleSpeed,
      u_RightWheelAngleSpeed,
    };
    matrixVectorProduct((float*)matrixB, us, stateVariablesEstimated, NUM_STATE_VARIABLES, NUM_CONTROL_INPUTS);
  
    float e1 = bodyAngleInRad - stateVariablesPrevious[0];
    float e2 = bodyAngleSpeedInRadPerSec - stateVariablesPrevious[1];
    float e3 = leftWheelAngleInRad - stateVariablesPrevious[2];
    float e4 = leftWheelAngleSpeedInRadPerSec - stateVariablesPrevious[3];
    float e5 = rightWheelAngleInRad - stateVariablesPrevious[4];
    float e6 = rightWheelAngleSpeedInRadPerSec - stateVariablesPrevious[5];
    float errors[NUM_OBSERVABLE_VARIABLES] = {
      e1,
      e2,
      e3,
      e4,
      e5,
      e6,
    };
    matrixVectorProduct((float*)observerGainMatrix, errors, stateVariablesEstimated, NUM_STATE_VARIABLES, NUM_OBSERVABLE_VARIABLES);

    // clip to the limit value.
    // wheelAngleSpeedInRadPerSecEstimated = constrain(wheelAngleSpeedInRadPerSecEstimated, -20*2*PI, 20*2*PI);

    copyArray(stateVariablesEstimated, stateVariablesPrevious, NUM_STATE_VARIABLES);

    // calc input
    u_LeftWheelAngleSpeed = -1.0 * dotProduct(feedbackGainsLeft, stateVariablesEstimated, NUM_STATE_VARIABLES);
    u_RightWheelAngleSpeed = -1.0 * dotProduct(feedbackGainsRight, stateVariablesEstimated, NUM_STATE_VARIABLES);
    u_LeftWheelAngleSpeed = constrain(u_LeftWheelAngleSpeed, -10*2*PI, 10*2*PI);
    u_RightWheelAngleSpeed = constrain(u_RightWheelAngleSpeed, -10*2*PI, 10*2*PI);

    Serial.print("  ## u_LeftWheelAngleSpeed: ");
    Serial.print(u_LeftWheelAngleSpeed);
    Serial.print("  ## u_RightWheelAngleSpeed: ");
    Serial.print(u_RightWheelAngleSpeed);

    float eA = u_LeftWheelAngleSpeed - leftWheelAngleSpeedInRadPerSec;
    float eB = u_RightWheelAngleSpeed - rightWheelAngleSpeedInRadPerSec;
    eAI = eA * controlTimeDeltaSec + eAI * (1.00 - 0.01);
    eBI = eB * controlTimeDeltaSec + eBI * (1.00 - 0.01);
    float uA = kpA * eA + kiA * eAI;
    float uB = kpB * eB + kiB * eBI;
    uA = constrain(uA, -1.0, 1.0);
    uB = constrain(uB, -1.0, 1.0);
    setLeftDutyCycle(uA);
    setRightDutyCycle(uB);

    Serial.print("  BodyAngle: ");
    Serial.print(bodyAngleInRad);
    Serial.println();

    readSensorFlag = false;
    angleSpeedAverageA = 0;
    angleSpeedAverageB = 0;
  }
}
