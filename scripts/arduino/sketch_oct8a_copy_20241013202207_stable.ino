// 1 wheel model with considering of the input delay.

#define sign(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : 0))
// #include "Servo.h"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
// #include <TimerOne.h>  // TimerOneライブラリをインクルード
#include <MsTimer2.h>

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

float T_sample = 0.010;  // sample width [sec]

float r_wheel = 2.7 * 0.01;  // wheel radius [m]
float g = 9.8;               // gravity acceleration [m/s^2]
float m_body = 250 * 0.001;  // body mass [kg]
float h_body = 10.5 * 0.01;   // body height between wheel axis and body mass center [m]

float m_wheel = 10.0 * 0.001;  // [kg]

float T_pendulum = 0.75;  // 振り子周期 [s]
float I_eff = pow(T_pendulum, 2) * m_body * g * h_body / (4 * pow(PI, 2));  // effective moment of inertia [?]

float tau_motor = 0.07;  // motor time consistency [s]

// float gainBodyAngle = -580.53314;
// float gainBodyAngleSpeed = -69.299985;
// float gainWheelAngle = -0.744813;
// float gainWheelAngleSpeed = -8.487374;
// float gainZ1 = 1.029974;
// float gainZ2 = 0.918933;
// float gainZ3 = 0.83972;
// float gainZ4 = 0.790838;
// float gainZ5 = 0.765403;
// float gainZ6 = 0.747254;
// float gainBodyAngle = -596.938515;
// float gainBodyAngleSpeed = -70.704009;
// float gainWheelAngle = -0.743052;
// float gainWheelAngleSpeed = -8.609511;
// float gainZ1 = 1.061461;
// float gainZ2 = 0.946159;
// float gainZ3 = 0.862297;
// float gainZ4 = 0.808284;
// float gainZ5 = 0.777281;
// float gainZ6 = 0.75336;
// float gainBodyAngle = -293.15151;
// float gainBodyAngleSpeed = -34.990785;
// float gainWheelAngle = -0.091897;
// float gainWheelAngleSpeed = -4.270842;
// float gainZ1 = 0.52113;
// float gainZ2 = 0.457118;
// float gainZ3 = 0.40093;
// float gainZ4 = 0.3516;
// float gainZ5 = 0.308284;
// float gainZ6 = 0.270248;
float gainBodyAngle = -378.545761;
float gainBodyAngleSpeed = -45.193282;
float gainWheelAngle = -0.901474;
float gainWheelAngleSpeed = -5.546318;
float gainZ1 = 0.667347;
float gainZ2 = 0.585332;
float gainZ3 = 0.5118;
float gainZ4 = 0.445059;
float gainZ5 = 0.38412;
float gainZ6 = 0.328582;
// discrete LQR gain K_d: [[-580.53314   -69.299985   -0.744813   -8.487374    1.029974    0.918933 0.83972     0.790838    0.765403    0.747254]]
// discrete LQR gain K_d: [[-596.938515  -70.704009   -0.743052   -8.609511    1.061461    0.946159 0.862297    0.808284    0.777281    0.75336 ]]
// discrete LQR gain K_d: [[-293.15151   -34.990785   -0.091897   -4.270842    0.52113     0.457118 0.40093     0.3516      0.308284    0.270248]]
// discrete LQR gain K_d: [[-296.095841  -35.460912   -0.091702   -4.322482    0.530754    0.467032 0.410816    0.360911    0.316429    0.276751]]
// discrete LQR gain K_d: [[-289.409316  -34.546899   -0.009196   -4.211365    0.515777    0.452581 0.397125    0.348457    0.305745    0.268259]]
// discrete LQR gain K_d: [[-299.471623  -35.747954   -0.290355   -4.377451    0.529023    0.463627 0.406241    0.355879    0.31168     0.27289 ]]
// discrete LQR gain K_d: [[-670.218275  -80.045819   -2.685098   -9.801257    1.192125    1.046127 0.905788    0.767399    0.631869    0.503193]]
// discrete LQR gain K_d: [[-378.545761  -45.193282   -0.901474   -5.546318    0.667347    0.585332 0.5118      0.445059    0.38412     0.328582]]

float z1Estimated = 0;
float z2Estimated = 0;
float z3Estimated = 0;
float z4Estimated = 0;
float z5Estimated = 0;
float z6Estimated = 0;
float z1EstimatedPrev = 0;
float z2EstimatedPrev = 0;
float z3EstimatedPrev = 0;
float z4EstimatedPrev = 0;
float z5EstimatedPrev = 0;
float z6EstimatedPrev = 0;

float anglePrevA = 0;
float angleSpeedA = 0;
float anglePrevB = 0;
float angleSpeedB = 0;
float timeNow1 = 0;
float timePrev1 = 0;
float timeNow2 = 0;
float timePrev2 = 0;

float angleA = 0;
float angleB = 0;

int timeDeltaMilli = 10;
float controlTimeDeltaSec = 0.010;
float angleSensorTimeDeltaSec = 0.0001;

float bodyAngleOffset = 0;

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
    // float lag = -3.4;
    sensors_event_t orientation_event;
    bno.getEvent(&orientation_event, Adafruit_BNO055::VECTOR_EULER);
    prev_value = current_value;
    if (abs(orientation_event.orientation.z) < 20) {
      return (prev_value - bodyAngleOffset) * PI / 180;  // 22
    };
    // Serial.print("orientation: ");
    // Serial.print(orientation_event.orientation.z);
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
// WheelAngleSpeedSampler wheelAngleSpeedSampler;

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

  // for (int i=0;i<250;i++) {
  //   sensors_event_t orientation_event;
  //   bno.getEvent(&orientation_event, Adafruit_BNO055::VECTOR_EULER);
  //   Serial.println(orientation_event.orientation.z);
  //   delay(20);
  // }
  int angleCalibrationCnt = 0; 
  for (int i=0;i<300;i++) {
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
  // bodyAngleOffset = -87.0;

  // // Timer1を設定。10msごとに割り込みを発生させる
  // Timer1.initialize(10000);  // 10000マイクロ秒（= 10ms）
  // // 割り込み関数を登録
  // Timer1.attachInterrupt(runController);

  // MsTimer2の設定
  MsTimer2::set(10, runController);  // タイマー割り込みの間隔とコールバック関数を設定
  MsTimer2::start();  // タイマー割り込みを開始
}


void runController() {
  // ここではフラグだけをセットし、実際のセンサー読み取りはloop()で行う
  readSensorFlag = true;
}

float bodyAngleInRadEstimated = 0;
float bodyAngleSpeedInRadPerSecEstimated = 0;
float wheelAngleInRadEstimated = 0;
float wheelAngleSpeedInRadPerSecEstimated = 0;

float bodyAngleInRadEstimatedPrev = 0;
float bodyAngleSpeedInRadPerSecEstimatedPrev = 0;
float wheelAngleInRadEstimatedPrev = 0;
float wheelAngleSpeedInRadPerSecEstimatedPrev = 0;

float u_WheelAngleSpeed = 0;
// float u_DutyMilliseconds = 0;
float u_pwm_value = 0;
float wheelAngleInRad = 0;

float angleSpeedAverageA = 0;
float angleSpeedAverageB = 0;


float kpA = 0.025;
float kpB = 0.03;
float kiA = 0.04;
float kiB = 0.05;
float eAI = 0.0;
float eBI = 0.0;
float alpha = 0.8;


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
    // wheelAngleInRad = angleB;
    // wheelAngleSpeedInRadPerSec = angleSpeedB;
    wheelAngleInRad = (angleA + angleB) / 2.0;
    // wheelAngleSpeedInRadPerSec = (angleSpeedA + angleSpeedB) / 2.0;
    wheelAngleSpeedInRadPerSec = (angleSpeedAverageA + angleSpeedAverageB) / 2.0;
 
    // wheelAngleSpeedInRadPerSec = wheelAngleSpeedSampler.sampleInRadPerSec();

    // start controling
    bodyAngleInRadEstimated = 0;
    bodyAngleSpeedInRadPerSecEstimated = 0;
    wheelAngleInRadEstimated = 0;
    wheelAngleSpeedInRadPerSecEstimated = 0;
    z1Estimated = 0;
    z2Estimated = 0;
    z3Estimated = 0;
    z4Estimated = 0;
    z5Estimated = 0;
    z6Estimated = 0;

    bodyAngleInRadEstimated += 1.003511 * bodyAngleInRadEstimatedPrev
      + 0.010012 * bodyAngleSpeedInRadPerSecEstimatedPrev
      + 0.000132 * wheelAngleSpeedInRadPerSecEstimatedPrev
      + -0.000109 * z1EstimatedPrev
      + -0.00002 * z2EstimatedPrev
      + -0.000003 * z3EstimatedPrev
    ;
    bodyAngleSpeedInRadPerSecEstimated += 0.70266 * bodyAngleInRadEstimatedPrev
      + 1.003511 * bodyAngleSpeedInRadPerSecEstimatedPrev
      + 0.025772 * wheelAngleSpeedInRadPerSecEstimatedPrev
      + -0.019247 * z1EstimatedPrev
      + -0.005332 * z2EstimatedPrev
      + -0.001024 * z3EstimatedPrev
      + -0.00015 * z4EstimatedPrev
      + -0.000018 * z5EstimatedPrev
      + -0.000002 * z6EstimatedPrev
    ;
    wheelAngleInRadEstimated += 1.0 * wheelAngleInRadEstimatedPrev
      + 0.009319 * wheelAngleSpeedInRadPerSecEstimatedPrev
      + 0.000562 * z1EstimatedPrev
      + 0.000103 * z2EstimatedPrev
      + 0.000015 * z3EstimatedPrev
      + 0.000002 * z4EstimatedPrev
      + 0.000000 * z5EstimatedPrev
      + 0.000000 * z6EstimatedPrev
    ;
    wheelAngleSpeedInRadPerSecEstimated += 0 * bodyAngleInRadEstimatedPrev
      + 0 * bodyAngleSpeedInRadPerSecEstimatedPrev
      + 0.866878 * wheelAngleSpeedInRadPerSecEstimatedPrev
      + 0.099396 * z1EstimatedPrev
      + 0.027555 * z2EstimatedPrev
      + 0.005295 * z3EstimatedPrev
      + 0.000775 * z4EstimatedPrev
      + 0.000092 * z5EstimatedPrev
      + 0.000009 * z6EstimatedPrev
    ;
    z1Estimated +=
      0.548812 * z1EstimatedPrev
      + 0.329287 * z1EstimatedPrev
      + 0.098786 * z1EstimatedPrev
      + 0.019757 * z1EstimatedPrev
      + 0.002964 * z1EstimatedPrev
      + 0.000356 * z1EstimatedPrev
    ;
    z2Estimated +=
      0.548812 * z2EstimatedPrev
      + 0.329287 * z3EstimatedPrev
      + 0.098786 * z4EstimatedPrev
      + 0.019757 * z5EstimatedPrev
      + 0.002964 * z6EstimatedPrev
    ;
    z3Estimated +=
      0.548812 * z3EstimatedPrev
      + 0.329287 * z4EstimatedPrev
      + 0.098786 * z5EstimatedPrev
      + 0.019757 * z6EstimatedPrev
    ;
    z4Estimated +=
      0.548812 * z4EstimatedPrev
      + 0.329287 * z5EstimatedPrev
      + 0.098786 * z6EstimatedPrev
    ;
    z5Estimated +=
      0.548812 * z5EstimatedPrev
      + 0.329287 * z6EstimatedPrev
    ;
    z6Estimated +=
      0.548812 * z6EstimatedPrev
    ;      
    // A_d (scipy.cont2discreteによる):
    // [[ 1.003511  0.010012  0.        0.000132 -0.000109 -0.00002  -0.000003  -0.       -0.       -0.      ]
    //  [ 0.70266   1.003511  0.        0.025772 -0.019247 -0.005332 -0.001024  -0.00015  -0.000018 -0.000002]
    //  [ 0.        0.        1.        0.009319  0.000562  0.000103  0.000015  0.000002  0.        0.      ]
    //  [ 0.        0.        0.        0.866878  0.099396  0.027555  0.005295  0.000775  0.000092  0.000009]
    //  [ 0.        0.        0.        0.        0.548812  0.329287  0.098786  0.019757  0.002964  0.000356]
    //  [ 0.        0.        0.        0.        0.        0.548812  0.329287  0.098786  0.019757  0.002964]
    //  [ 0.        0.        0.        0.        0.        0.        0.548812  0.329287  0.098786  0.019757]
    //  [ 0.        0.        0.        0.        0.        0.        0.        0.548812  0.329287  0.098786]
    //  [ 0.        0.        0.        0.        0.        0.        0.        0.        0.548812  0.329287]
    //  [ 0.        0.        0.        0.        0.        0.        0.        0.        0.        0.548812]]

    bodyAngleInRadEstimated += -0 * u_WheelAngleSpeed;
    bodyAngleSpeedInRadPerSecEstimated += -0 * u_WheelAngleSpeed;
    wheelAngleInRadEstimated += 0 * u_WheelAngleSpeed;
    wheelAngleSpeedInRadPerSecEstimated += 0.000001 * u_WheelAngleSpeed;
    z1Estimated += 0.000039 * u_WheelAngleSpeed;
    z2Estimated += 0.00039 * u_WheelAngleSpeed;
    z3Estimated += 0.003358 * u_WheelAngleSpeed;
    z4Estimated += 0.023115 * u_WheelAngleSpeed;
    z5Estimated += 0.121901 * u_WheelAngleSpeed;
    z6Estimated += 0.451188 * u_WheelAngleSpeed;
    // B_d (scipy.cont2discreteによる):
    // [[-0.      ]
    //  [-0.      ]
    //  [ 0.      ]
    //  [ 0.000001]
    //  [ 0.000039]
    //  [ 0.000394]
    //  [ 0.003358]
    //  [ 0.023115]
    //  [ 0.121901]
    //  [ 0.451188]]

    float e1 = bodyAngleInRad - bodyAngleInRadEstimatedPrev;
    float e2 = bodyAngleSpeedInRadPerSec - bodyAngleSpeedInRadPerSecEstimatedPrev;
    float e3 = wheelAngleInRad - wheelAngleInRadEstimatedPrev;
    float e4 = wheelAngleSpeedInRadPerSec - wheelAngleSpeedInRadPerSecEstimatedPrev;
    bodyAngleInRadEstimated += 0.6046 * e1 + 0.064392 * e2 + 0.000002 * e3 + 0.000048 * e4;
    bodyAngleSpeedInRadPerSecEstimated += 0.064392 * e1 + 0.662611 * e2 + 0.00001 * e3 + 0.000278 * e4;
    wheelAngleInRadEstimated += -0.000002 * e1 + 0.00001 * e2 + 0.618044 * e3 + 0.000941 * e4;
    wheelAngleSpeedInRadPerSecEstimated += 0.000048 * e1 + 0.000278 * e2 + 0.000941 * e3 + 0.602982 * e4;
    z1Estimated += 0.005301 * e1 -0.020367 * e2 + 0.0008 * e3 + 0.137743 * e4;
    z2Estimated += 0.003264 * e1 -0.012777 * e2 + 0.000452 * e3 + 0.085731 * e4;
    z3Estimated += 0.001733 * e1 -0.006923 * e2 + 0.00023 * e3 + 0.046081 * e4;
    z4Estimated += 0.00076 * e1 -0.003108 * e2 + 0.000096 * e3 + 0.02051 * e4;
    z5Estimated += 0.000247 * e1 -0.001037 * e2 + 0.00003 * e3 + 0.006781 * e4;
    z6Estimated += 0.000045 * e1 -0.000195 * e2 + 0.000005 * e3 + 0.00126 * e4;
    // オブザーバゲイン L: [[ 0.6046    0.064392 -0.000002  0.000048]
    //  [ 0.064392  0.662611  0.00001   0.000278]
    //  [-0.000002  0.00001   0.618044  0.000941]
    //  [ 0.000048  0.000278  0.000941  0.602982]
    //  [ 0.005301 -0.020367  0.0008    0.137743]
    //  [ 0.003264 -0.012777  0.000452  0.085731]
    //  [ 0.001733 -0.006923  0.00023   0.046081]
    //  [ 0.00076  -0.003108  0.000096  0.02051 ]
    //  [ 0.000247 -0.001037  0.00003   0.006781]
    //  [ 0.000045 -0.000195  0.000005  0.00126 ]]

    // clip to the limit value.
    wheelAngleSpeedInRadPerSecEstimated = constrain(wheelAngleSpeedInRadPerSecEstimated, -20*2*PI, 20*2*PI);

    bodyAngleInRadEstimatedPrev = bodyAngleInRadEstimated;
    bodyAngleSpeedInRadPerSecEstimatedPrev = bodyAngleSpeedInRadPerSecEstimated;
    wheelAngleInRadEstimatedPrev = wheelAngleInRadEstimated;
    wheelAngleSpeedInRadPerSecEstimatedPrev = wheelAngleSpeedInRadPerSecEstimated;
    z1EstimatedPrev = z1Estimated;
    z2EstimatedPrev = z2Estimated;
    z3EstimatedPrev = z3Estimated;
    z4EstimatedPrev = z4Estimated;
    z5EstimatedPrev = z5Estimated;
    z6EstimatedPrev = z6Estimated;

    // calc input
    u_WheelAngleSpeed =
      - gainBodyAngle * bodyAngleInRadEstimated
      //- gainBodyAngle * bodyAngleInRad
      - gainBodyAngleSpeed * bodyAngleSpeedInRadPerSecEstimated
      //- gainBodyAngleSpeed * bodyAngleSpeedInRadPerSec
      // - gainBodySpeed * bodySpeedObserved
      - gainWheelAngle * wheelAngleInRadEstimated;
      - gainWheelAngleSpeed * wheelAngleSpeedInRadPerSecEstimated;
    Serial.print("  ## u_WheelAngleSpeed: ");                                                      // ロール角
    Serial.print(u_WheelAngleSpeed);                                                            // ロール角

    float eA = u_WheelAngleSpeed - angleSpeedAverageA + 0.0*(angleSpeedAverageB - angleSpeedAverageA) + 1.0*(angleB - angleA);
    float eB = u_WheelAngleSpeed - angleSpeedAverageB + 0.0*(angleSpeedAverageA - angleSpeedAverageB) + 1.0*(angleA - angleB);
    // float eA = u_WheelAngleSpeed - angleSpeedA;
    // float eB = u_WheelAngleSpeed - angleSpeedB;
    eAI = eA * controlTimeDeltaSec + eAI * (1.00 - 0.01);
    eBI = eB * controlTimeDeltaSec + eBI * (1.00 - 0.01);
    float uA = kpA * eA + kiA * eAI;
    float uB = kpB * eB + kiB * eBI;
    uA = constrain(uA * 1.0, -1.0, 1.0);
    uB = constrain(uB * 1.0, -1.0, 1.0);
    setLeftDutyCycle(uA); 
    setRightDutyCycle(uB);

    Serial.print("  BodyAngle: ");                                                          // ロール角
    Serial.print(bodyAngleInRad);                                                         // ロール角
    // // Serial.print("  BodyAngleSpeed: ");                                                   // ロール角
    // // Serial.print(bodyAngleSpeedInRadPerSec);                                              // ロール角
    // Serial.print("  u_WheelAngleSpeed: ");                                                      // ロール角
    // Serial.print(u_WheelAngleSpeed);                                                            // ロール角
    // Serial.print("  u_DutyMilliseconds: ");                                               // ロール角
    // Serial.print(u_DutyMilliseconds);  // ロール角
    // Serial.print("  uA: ");                                               // ロール角
    // Serial.print(uA);  // ロール角
    // Serial.print("  uB: ");                                               // ロール角
    // Serial.print(uB);  // ロール角
    // // // // Serial.println("");  // ロール角

    // Serial.print(" bodyAngleInRadEstimated: ");                                                          // ロール角
    // Serial.print(bodyAngleInRadEstimated);                                                         // ロール角
    // // Serial.print("  bodyAngleSpeedInRadPerSecEstimated: ");                                                   // ロール角
    // // Serial.print(bodyAngleSpeedInRadPerSecEstimated);                                              // ロール角
    // // // Serial.print("  bodySpeedObserved: ");                                                      // ロール角
    // // // Serial.print(bodySpeedObserved * 1000);                                                            // ロール角
    // Serial.print("  wheelAngleInRad: ");                                               // ロール角
    // Serial.print(wheelAngleInRad);  // ロール角
    // Serial.print("  wheelAngleSpeedInRadPerSec: ");                                               // ロール角
    // Serial.print(wheelAngleSpeedInRadPerSec);  // ロール角
    // Serial.print("  wheelAngleSpeedInRadPerSecEstimated: ");                                               // ロール角
    // Serial.println(wheelAngleSpeedInRadPerSecEstimated);  // ロール角

    Serial.println();

    readSensorFlag = false;
    angleSpeedAverageA = 0;
    angleSpeedAverageB = 0;
  }
}
