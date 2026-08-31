/*
 * agrix_motor_mega.ino  —  Agri-X 하위 제어기 (Arduino Mega 2560)
 * ---------------------------------------------------------------
 * 역할 분담
 *   상위(라즈베리파이 5)  : 비전/ToF/마커/FSM. "좌우 바퀴를 몇 m/s 로 돌려라"만 내려보낸다.
 *   하위(이 스케치, Mega) : 엔코더를 읽어 **속도 PID** 로 그 명령을 추종한다.
 *
 * 왜 나누는가
 *   Pi 는 리눅스라 수십 ms 단위 지터가 있어 100Hz 속도 루프를 안정적으로 돌리기 어렵다.
 *   엔코더 인터럽트도 Pi 의 파이썬 콜백으로는 카운트를 놓친다(13PPR x 4체배 x 감속비).
 *   Mega 는 지터 없이 100Hz 로 돌고 인터럽트를 놓치지 않는다.
 *
 * 엔코더
 *   13 PPR, A/B 쿼드러처, 4체배 -> 모터축 1회전당 52 카운트.
 *   바퀴축 카운트 = 52 * GEAR_RATIO.  ★ GEAR_RATIO 는 반드시 실측해서 넣을 것.
 *   Mega 의 외부 인터럽트 핀: 2,3,18,19,20,21 (20/21 은 I2C 와 겸용이므로 피함)
 *
 * 시리얼 프로토콜 (USB, 115200)  — 줄 단위 ASCII
 *   Pi -> Mega
 *     "V <left_mps> <right_mps>\n"   속도 명령 (m/s, 부호 = 전/후진)
 *     "S\n"                          즉시 정지
 *     "P <kp> <ki> <kd>\n"           PID 게인 변경 (현장 튜닝용)
 *     "?\n"                          상태 1회 요청
 *   Mega -> Pi  (기본 20Hz 자동 송신)
 *     "T <l_mps> <r_mps> <l_ticks> <r_ticks> <l_duty> <r_duty> <flags>\n"
 *     flags bit0 = 워치독 정지 상태
 *
 * 안전
 *   WATCHDOG_MS 안에 새 명령이 없으면 **스스로 정지**한다.
 *   USB 케이블이 빠지거나 Pi 가 죽어도 로봇이 계속 달리지 않게 하는 마지막 방어선이다.
 */

#include <Arduino.h>

// ---------------- 핀맵 (★ 실제 배선에 맞게 수정) ----------------
// 모터 드라이버 (BTS7960 / L298N 등 PWM+DIR 방식 가정)
const uint8_t PIN_L_PWM = 9;
const uint8_t PIN_L_DIR = 8;
const uint8_t PIN_R_PWM = 10;
const uint8_t PIN_R_DIR = 11;

// 엔코더 A/B  (A 는 반드시 인터럽트 핀)
const uint8_t PIN_ENC_L_A = 2;    // INT0
const uint8_t PIN_ENC_L_B = 4;
const uint8_t PIN_ENC_R_A = 3;    // INT1
const uint8_t PIN_ENC_R_B = 5;

// ---------------- 기구 파라미터 (★ 실측) ----------------
const float PPR          = 13.0f;   // 엔코더 pulse/rev (모터축)
const float DECODE       = 4.0f;    // 4체배
const float GEAR_RATIO   = 1.0f;    // ★ 모터축 -> 바퀴축 감속비. 반드시 실측
const float WHEEL_RADIUS = 0.033f;  // m (궤도 두께 포함 유효 반지름)
const float TICKS_PER_WHEEL_REV = PPR * DECODE * GEAR_RATIO;
const float METERS_PER_TICK = (2.0f * PI * WHEEL_RADIUS) / TICKS_PER_WHEEL_REV;

// ---------------- 제어 파라미터 ----------------
const uint16_t CONTROL_HZ   = 100;
const uint16_t CONTROL_MS   = 1000 / CONTROL_HZ;
const uint16_t TELEM_MS     = 50;     // 20Hz 텔레메트리
const uint16_t WATCHDOG_MS  = 400;    // 이 시간 무명령 -> 정지
const float MAX_MPS         = 0.60f;  // ★ 실측: 듀티 100% 일 때의 바퀴 선속도
const uint8_t MIN_DUTY      = 40;     // 데드밴드 보상(0~255). ★ 실측
float Kp = 220.0f, Ki = 900.0f, Kd = 0.0f;   // 출력 단위 = PWM count
const float I_LIMIT = 200.0f;

// ---------------- 엔코더 상태 ----------------
volatile long encL = 0, encR = 0;

// A 상승 에지에서 B 레벨로 방향 판정(2체배 상당). 4체배가 필요하면
// B 핀도 인터럽트 핀에 물리고 아래 ISR 을 하나 더 붙이면 된다.
void isrLeftA()  { encL += (digitalRead(PIN_ENC_L_B) ? 1 : -1); }
void isrRightA() { encR += (digitalRead(PIN_ENC_R_B) ? 1 : -1); }

// ---------------- PID 상태 ----------------
struct Wheel {
  float target = 0.0f;      // m/s
  float measured = 0.0f;    // m/s
  float integral = 0.0f;
  float prevErr = 0.0f;
  int   duty = 0;           // -255 ~ 255
  long  lastTicks = 0;
};
Wheel wl, wr;

bool watchdogStopped = false;
unsigned long lastCmdMs = 0, lastCtlMs = 0, lastTelemMs = 0;

// ---------------- 저수준 출력 ----------------
void applyDuty(uint8_t pinPwm, uint8_t pinDir, int duty) {
  bool forward = (duty >= 0);
  int mag = abs(duty);
  if (mag < 3) {                 // 완전 정지 (덜덜거림 방지)
    digitalWrite(pinDir, LOW);
    analogWrite(pinPwm, 0);
    return;
  }
  // 데드밴드 보상: MIN_DUTY ~ 255 구간으로 매핑
  mag = MIN_DUTY + (int)((255 - MIN_DUTY) * (mag / 255.0f));
  if (mag > 255) mag = 255;
  digitalWrite(pinDir, forward ? HIGH : LOW);
  analogWrite(pinPwm, mag);
}

void stopAll() {
  wl.target = wr.target = 0.0f;
  wl.integral = wr.integral = 0.0f;
  wl.duty = wr.duty = 0;
  applyDuty(PIN_L_PWM, PIN_L_DIR, 0);
  applyDuty(PIN_R_PWM, PIN_R_DIR, 0);
}

// ---------------- PID 한 스텝 ----------------
void stepWheel(Wheel &w, long ticksNow, float dt, uint8_t pinPwm, uint8_t pinDir) {
  long d = ticksNow - w.lastTicks;
  w.lastTicks = ticksNow;
  w.measured = (d * METERS_PER_TICK) / dt;

  float err = w.target - w.measured;
  w.integral += err * dt;
  if (w.integral >  I_LIMIT) w.integral =  I_LIMIT;
  if (w.integral < -I_LIMIT) w.integral = -I_LIMIT;
  float deriv = (dt > 0.0f) ? (err - w.prevErr) / dt : 0.0f;
  w.prevErr = err;

  // 피드포워드(명령 자체) + PI(D) 보정
  float ff = (w.target / MAX_MPS) * 255.0f;
  float out = ff + Kp * err + Ki * w.integral + Kd * deriv;
  if (out >  255.0f) out =  255.0f;
  if (out < -255.0f) out = -255.0f;

  // 목표가 0 이면 적분 잔량으로 기어가지 않게 확실히 끊는다
  if (fabs(w.target) < 1e-4f) { out = 0.0f; w.integral = 0.0f; }

  w.duty = (int)out;
  applyDuty(pinPwm, pinDir, w.duty);
}

// ---------------- 시리얼 명령 파싱 ----------------
char buf[64];
uint8_t buflen = 0;

void handleLine(char *line) {
  if (line[0] == 'V') {
    float l = 0, r = 0;
    if (sscanf(line + 1, "%f %f", &l, &r) == 2) {
      if (l >  MAX_MPS) l =  MAX_MPS;
      if (l < -MAX_MPS) l = -MAX_MPS;
      if (r >  MAX_MPS) r =  MAX_MPS;
      if (r < -MAX_MPS) r = -MAX_MPS;
      wl.target = l; wr.target = r;
      lastCmdMs = millis();
      watchdogStopped = false;
    }
  } else if (line[0] == 'S') {
    stopAll();
    lastCmdMs = millis();
  } else if (line[0] == 'P') {
    float a, b, c;
    if (sscanf(line + 1, "%f %f %f", &a, &b, &c) == 3) { Kp = a; Ki = b; Kd = c; }
  } else if (line[0] == '?') {
    lastTelemMs = 0;   // 즉시 1회 송신
  }
}

void pollSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (buflen) { buf[buflen] = '\0'; handleLine(buf); buflen = 0; }
    } else if (buflen < sizeof(buf) - 1) {
      buf[buflen++] = c;
    }
  }
}

// ---------------- 설정 / 루프 ----------------
void setup() {
  Serial.begin(115200);
  pinMode(PIN_L_PWM, OUTPUT); pinMode(PIN_L_DIR, OUTPUT);
  pinMode(PIN_R_PWM, OUTPUT); pinMode(PIN_R_DIR, OUTPUT);
  pinMode(PIN_ENC_L_A, INPUT_PULLUP); pinMode(PIN_ENC_L_B, INPUT_PULLUP);
  pinMode(PIN_ENC_R_A, INPUT_PULLUP); pinMode(PIN_ENC_R_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_L_A), isrLeftA,  RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_R_A), isrRightA, RISING);
  stopAll();
  lastCmdMs = lastCtlMs = lastTelemMs = millis();
  Serial.println(F("# agrix mega ready"));
}

void loop() {
  pollSerial();
  unsigned long now = millis();

  // --- 워치독: 상위가 조용하면 스스로 멈춘다 ---
  if (now - lastCmdMs > WATCHDOG_MS && !watchdogStopped) {
    stopAll();
    watchdogStopped = true;
  }

  // --- 속도 PID (100Hz) ---
  if (now - lastCtlMs >= CONTROL_MS) {
    float dt = (now - lastCtlMs) / 1000.0f;
    lastCtlMs = now;
    long l, r;
    noInterrupts(); l = encL; r = encR; interrupts();
    stepWheel(wl, l, dt, PIN_L_PWM, PIN_L_DIR);
    stepWheel(wr, r, dt, PIN_R_PWM, PIN_R_DIR);
  }

  // --- 텔레메트리 (20Hz) ---
  if (now - lastTelemMs >= TELEM_MS) {
    lastTelemMs = now;
    long l, r;
    noInterrupts(); l = encL; r = encR; interrupts();
    Serial.print(F("T "));
    Serial.print(wl.measured, 4); Serial.print(' ');
    Serial.print(wr.measured, 4); Serial.print(' ');
    Serial.print(l);              Serial.print(' ');
    Serial.print(r);              Serial.print(' ');
    Serial.print(wl.duty);        Serial.print(' ');
    Serial.print(wr.duty);        Serial.print(' ');
    Serial.println(watchdogStopped ? 1 : 0);
  }
}
