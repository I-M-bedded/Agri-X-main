#include <Arduino.h>
#include <util/atomic.h>

/*
 * Agri-X Mega Motion V2 - USB hybrid velocity/position controller
 *
 * Pi -> Mega (115200 baud, newline-delimited ASCII)
 *   DRIVE <left_rpm> <right_rpm>              continuous navigation
 *   MOVE <seq> <left_deg> <right_deg> <rpm>   finite encoder move
 *   HB | STOP | STATUS | PING
 *
 * Mega -> Pi
 *   STATE <seq> <mode> <left_deg> <right_deg> <left_rpm> <right_rpm> (20 Hz)
 *   ACK <seq> | DONE <seq> | STOPPED | PONG | ERR <reason>
 *
 * DRIVE must be refreshed by the Pi control loop and times out after 400 ms.
 * HB refreshes only a blocking MOVE. STATUS/PING are diagnostic commands and
 * deliberately do not refresh either motion watchdog. USB disconnect therefore
 * stops either mode without depending on Linux or Python cleanup.
 */

/* BTS7960 pin map: keep R_EN/L_EN high on each driver. */
constexpr uint8_t MOTOR1_RPWM_PIN = 6;   // Timer4 OC4A
constexpr uint8_t MOTOR1_LPWM_PIN = 7;   // Timer4 OC4B
constexpr uint8_t MOTOR2_RPWM_PIN = 44;  // Timer5 OC5C
constexpr uint8_t MOTOR2_LPWM_PIN = 45;  // Timer5 OC5B

constexpr uint8_t ENCODER1_A_PIN = 2;
constexpr uint8_t ENCODER1_B_PIN = 3;
constexpr uint8_t ENCODER2_A_PIN = 18;
constexpr uint8_t ENCODER2_B_PIN = 19;

// JGB3865-520R45-12 assumption: 11 pulses/motor-rev * 45:1 * x4 decode.
constexpr float ENCODER1_CPR = 1980.0f;
constexpr float ENCODER2_CPR = 1980.0f;
constexpr int8_t ENCODER1_SIGN = 1;
constexpr int8_t ENCODER2_SIGN = 1;

// Positive logical wheel rotation means vehicle-forward rotation.
constexpr int8_t MOTOR1_FORWARD_SIGN = -1;
constexpr int8_t MOTOR2_FORWARD_SIGN = 1;

constexpr float MANUAL_RPM = 80.0f;
constexpr float MAX_COMMAND_RPM = 150.0f;
constexpr float MAX_DUTY = 0.45f;
constexpr float POSITION_KP_RPM_PER_DEG = 1.0f;
constexpr float POSITION_TOLERANCE_DEG = 2.0f;
constexpr float SETTLED_RPM = 3.0f;
constexpr uint16_t SETTLED_TIME_MS = 150U;
// DRIVE is refreshed by the Pi control loop. A short watchdog is the final
// safety layer for a dead process or unplugged USB cable.
constexpr uint32_t DRIVE_TIMEOUT_MS = 400UL;
constexpr uint32_t LINK_TIMEOUT_MS = 1000UL;
constexpr uint32_t CONTROL_PERIOD_US = 1000UL;
constexpr uint32_t POSITION_PERIOD_US = 10000UL;
constexpr uint16_t TELEMETRY_PERIOD_MS = 50U;
constexpr uint16_t PWM_TOP = 799U; // 16 MHz / (799 + 1) = 20 kHz

struct PidController {
  float kp;
  float ki;
  float kd;
  float integrator;
  float previousMeasurement;
  float derivativeState;
};

struct MotorAxis {
  PidController pid;
  float targetRpm;
  float measuredRpm;
  float duty;
  float encoderCpr;
  int32_t previousCount;
};

enum class ControlMode : uint8_t { IDLE, MANUAL, POSITION };

volatile int32_t encoderCount[2] = {0, 0};
volatile uint8_t encoderPrevious[2] = {0, 0};
MotorAxis motors[2];
ControlMode controlMode = ControlMode::IDLE;
bool motorsEnabled = false;
int32_t positionTarget[2] = {0, 0};
float moveMaxRpm = MANUAL_RPM;
uint32_t activeSequence = 0;
uint32_t lastDriveCommandMs = 0;
uint32_t lastPositionLinkMs = 0;
uint32_t nextControlUs = 0;
uint32_t nextPositionUs = 0;
uint32_t nextTelemetryMs = 0;
uint16_t settledTimeMs = 0;
char lineBuffer[96];
uint8_t lineLength = 0;

constexpr int8_t QUADRATURE_TRANSITION[16] = {
   0, -1,  1,  0,
   1,  0,  0, -1,
  -1,  0,  0,  1,
   0,  1, -1,  0
};

float clampFloat(float value, float low, float high) {
  if (value > high) return high;
  if (value < low) return low;
  return value;
}

void updateEncoder(uint8_t motor) {
  uint8_t current;
  int8_t sign;

  if (motor == 0) {
    current = static_cast<uint8_t>((digitalRead(ENCODER1_A_PIN) ? 1U : 0U)
                                  | (digitalRead(ENCODER1_B_PIN) ? 2U : 0U));
    sign = ENCODER1_SIGN;
  } else {
    current = static_cast<uint8_t>((digitalRead(ENCODER2_A_PIN) ? 1U : 0U)
                                  | (digitalRead(ENCODER2_B_PIN) ? 2U : 0U));
    sign = ENCODER2_SIGN;
  }

  encoderCount[motor] += sign * QUADRATURE_TRANSITION[(encoderPrevious[motor] << 2U) | current];
  encoderPrevious[motor] = current;
}

void encoder1AIsr() { updateEncoder(0); }
void encoder1BIsr() { updateEncoder(0); }
void encoder2AIsr() { updateEncoder(1); }
void encoder2BIsr() { updateEncoder(1); }

int32_t readEncoder(uint8_t motor) {
  int32_t value;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    value = encoderCount[motor];
  }
  return value;
}

void setupPwm20kHz() {
  pinMode(MOTOR1_RPWM_PIN, OUTPUT);
  pinMode(MOTOR1_LPWM_PIN, OUTPUT);
  pinMode(MOTOR2_RPWM_PIN, OUTPUT);
  pinMode(MOTOR2_LPWM_PIN, OUTPUT);

  // Timer4: mode 14 fast PWM, TOP=ICR4, no prescaler, D6/D7 outputs.
  TCCR4A = _BV(COM4A1) | _BV(COM4B1) | _BV(WGM41);
  TCCR4B = _BV(WGM43) | _BV(WGM42) | _BV(CS40);
  ICR4 = PWM_TOP;
  OCR4A = 0;
  OCR4B = 0;

  // Timer5: mode 14 fast PWM, TOP=ICR5, no prescaler, D44/D45 outputs.
  TCCR5A = _BV(COM5C1) | _BV(COM5B1) | _BV(WGM51);
  TCCR5B = _BV(WGM53) | _BV(WGM52) | _BV(CS50);
  ICR5 = PWM_TOP;
  OCR5C = 0;
  OCR5B = 0;
}

void setMotorDuty(uint8_t motor, float duty) {
  duty = clampFloat(duty, -MAX_DUTY, MAX_DUTY);
  const uint16_t compare = static_cast<uint16_t>(fabs(duty) * static_cast<float>(PWM_TOP + 1U));

  if (motor == 0) {
    OCR4A = duty > 0.0f ? compare : 0;
    OCR4B = duty < 0.0f ? compare : 0;
  } else {
    OCR5C = duty > 0.0f ? compare : 0;
    OCR5B = duty < 0.0f ? compare : 0;
  }
}

void resetPid(MotorAxis &motor) {
  motor.pid.integrator = 0.0f;
  motor.pid.previousMeasurement = motor.measuredRpm;
  motor.pid.derivativeState = 0.0f;
}

float pidStep(MotorAxis &motor, float dt) {
  const float error = motor.targetRpm - motor.measuredRpm;
  const float rawDerivative = -(motor.measuredRpm - motor.pid.previousMeasurement) / dt;
  motor.pid.derivativeState += 0.20f * (rawDerivative - motor.pid.derivativeState);

  const float unsaturated = motor.pid.kp * error + motor.pid.integrator
                          + motor.pid.kd * motor.pid.derivativeState;
  const float output = clampFloat(unsaturated, -MAX_DUTY, MAX_DUTY);
  motor.pid.integrator += (motor.pid.ki * error + 4.0f * (output - unsaturated)) * dt;
  motor.pid.integrator = clampFloat(motor.pid.integrator, -MAX_DUTY, MAX_DUTY);
  motor.pid.previousMeasurement = motor.measuredRpm;
  return output;
}

void controlStep() {
  constexpr float dt = 0.001f;

  for (uint8_t i = 0; i < 2; ++i) {
    MotorAxis &motor = motors[i];
    const int32_t count = readEncoder(i);
    const int32_t delta = count - motor.previousCount;
    const float rawRpm = (static_cast<float>(delta) * 60.0f) / (motor.encoderCpr * dt);
    motor.previousCount = count;
    motor.measuredRpm += 0.20f * (rawRpm - motor.measuredRpm);

    if (motorsEnabled) {
      motor.duty = pidStep(motor, dt);
    } else {
      motor.duty = 0.0f;
      resetPid(motor);
    }
    setMotorDuty(i, motor.duty);
  }
}

void stopMotors() {
  motors[0].targetRpm = 0.0f;
  motors[1].targetRpm = 0.0f;
  motorsEnabled = false;
  controlMode = ControlMode::IDLE;
  activeSequence = 0;
  settledTimeMs = 0;
  setMotorDuty(0, 0.0f);
  setMotorDuty(1, 0.0f);
}

void resetBothPid() {
  resetPid(motors[0]);
  resetPid(motors[1]);
}

void driveVelocity(float logicalLeftRpm, float logicalRightRpm) {
  logicalLeftRpm = clampFloat(logicalLeftRpm, -MAX_COMMAND_RPM, MAX_COMMAND_RPM);
  logicalRightRpm = clampFloat(logicalRightRpm, -MAX_COMMAND_RPM, MAX_COMMAND_RPM);

  if (fabs(logicalLeftRpm) < 0.01f && fabs(logicalRightRpm) < 0.01f) {
    stopMotors();
    lastDriveCommandMs = millis();
    return;
  }

  // Preserve PI state while the 20 Hz Pi loop refreshes DRIVE. Reset only
  // when changing into velocity mode, never on every command.
  if (controlMode != ControlMode::MANUAL || !motorsEnabled) {
    resetBothPid();
  }
  motors[0].targetRpm = logicalLeftRpm * MOTOR1_FORWARD_SIGN;
  motors[1].targetRpm = logicalRightRpm * MOTOR2_FORWARD_SIGN;
  motorsEnabled = true;
  controlMode = ControlMode::MANUAL;
  activeSequence = 0;
  lastDriveCommandMs = millis();
}

int32_t degreesToCounts(float degrees, float cpr, int8_t forwardSign) {
  const float raw = degrees * cpr * static_cast<float>(forwardSign) / 360.0f;
  return static_cast<int32_t>(raw >= 0.0f ? raw + 0.5f : raw - 0.5f);
}

void beginPositionMove(uint32_t sequence, float leftDegrees, float rightDegrees, float maxRpm) {
  positionTarget[0] = readEncoder(0) + degreesToCounts(leftDegrees, ENCODER1_CPR,
                                                       MOTOR1_FORWARD_SIGN);
  positionTarget[1] = readEncoder(1) + degreesToCounts(rightDegrees, ENCODER2_CPR,
                                                       MOTOR2_FORWARD_SIGN);
  moveMaxRpm = clampFloat(fabs(maxRpm), 5.0f, MAX_COMMAND_RPM);
  activeSequence = sequence;
  settledTimeMs = 0;
  resetBothPid();
  motorsEnabled = true;
  controlMode = ControlMode::POSITION;
  lastPositionLinkMs = millis();
  nextPositionUs = micros();
}

void positionStep() {
  bool insideTolerance = true;
  bool nearlyStopped = true;

  for (uint8_t i = 0; i < 2; ++i) {
    const int32_t countError = positionTarget[i] - readEncoder(i);
    const float errorDegrees = static_cast<float>(countError) * 360.0f / motors[i].encoderCpr;

    if (fabs(errorDegrees) <= POSITION_TOLERANCE_DEG) {
      motors[i].targetRpm = 0.0f;
    } else {
      motors[i].targetRpm = clampFloat(POSITION_KP_RPM_PER_DEG * errorDegrees,
                                       -moveMaxRpm, moveMaxRpm);
      insideTolerance = false;
    }
    if (fabs(motors[i].measuredRpm) > SETTLED_RPM) {
      nearlyStopped = false;
    }
  }

  if (insideTolerance && nearlyStopped) {
    settledTimeMs += POSITION_PERIOD_US / 1000UL;
    if (settledTimeMs >= SETTLED_TIME_MS) {
      const uint32_t completedSequence = activeSequence;
      stopMotors();
      Serial.print(F("DONE "));
      Serial.println(completedSequence);
    }
  } else {
    settledTimeMs = 0;
  }
}

const __FlashStringHelper *modeName() {
  switch (controlMode) {
    case ControlMode::MANUAL: return F("MANUAL");
    case ControlMode::POSITION: return F("POSITION");
    default: return F("IDLE");
  }
}

void printStatus() {
  const float leftDegrees = static_cast<float>(readEncoder(0)) * 360.0f
                          * MOTOR1_FORWARD_SIGN / ENCODER1_CPR;
  const float rightDegrees = static_cast<float>(readEncoder(1)) * 360.0f
                           * MOTOR2_FORWARD_SIGN / ENCODER2_CPR;
  const float leftRpm = motors[0].measuredRpm * MOTOR1_FORWARD_SIGN;
  const float rightRpm = motors[1].measuredRpm * MOTOR2_FORWARD_SIGN;

  Serial.print(F("STATE "));
  Serial.print(activeSequence);
  Serial.print(' ');
  Serial.print(modeName());
  Serial.print(' ');
  Serial.print(leftDegrees, 2);
  Serial.print(' ');
  Serial.print(rightDegrees, 2);
  Serial.print(' ');
  Serial.print(leftRpm, 2);
  Serial.print(' ');
  Serial.println(rightRpm, 2);
}

bool parseUnsigned(const char *text, uint32_t &value) {
  if (text == nullptr || *text == '\0' || *text == '-') return false;
  char *end = nullptr;
  const unsigned long parsed = strtoul(text, &end, 10);
  if (*end != '\0') return false;
  value = parsed;
  return true;
}

bool parseFloatValue(const char *text, float &value) {
  if (text == nullptr || *text == '\0') return false;
  char *end = nullptr;
  value = static_cast<float>(strtod(text, &end));
  return *end == '\0' && isfinite(value);
}

void processLine(char *line) {
  char *command = strtok(line, " \t");
  if (command == nullptr) return;

  if (strcmp(command, "MOVE") == 0) {
    char *sequenceText = strtok(nullptr, " \t");
    char *leftText = strtok(nullptr, " \t");
    char *rightText = strtok(nullptr, " \t");
    char *rpmText = strtok(nullptr, " \t");
    uint32_t sequence;
    float leftDegrees;
    float rightDegrees;
    float maxRpm;
    if (!parseUnsigned(sequenceText, sequence)
        || !parseFloatValue(leftText, leftDegrees)
        || !parseFloatValue(rightText, rightDegrees)
        || !parseFloatValue(rpmText, maxRpm)
        || strtok(nullptr, " \t") != nullptr
        || maxRpm <= 0.0f) {
      Serial.println(F("ERR BAD_MOVE"));
      return;
    }
    beginPositionMove(sequence, leftDegrees, rightDegrees, maxRpm);
    Serial.print(F("ACK "));
    Serial.println(sequence);
    return;
  }

  if (strcmp(command, "DRIVE") == 0) {
    char *leftText = strtok(nullptr, " \t");
    char *rightText = strtok(nullptr, " \t");
    float leftRpm;
    float rightRpm;
    if (!parseFloatValue(leftText, leftRpm)
        || !parseFloatValue(rightText, rightRpm)
        || strtok(nullptr, " \t") != nullptr) {
      Serial.println(F("ERR BAD_DRIVE"));
      return;
    }
    driveVelocity(leftRpm, rightRpm);
    return;
  }

  if (strcmp(command, "HB") == 0) {
    if (controlMode == ControlMode::POSITION) {
      lastPositionLinkMs = millis();
    }
    return;
  }

  if (strcmp(command, "STOP") == 0) {
    stopMotors();
    Serial.println(F("STOPPED"));
    return;
  }

  if (strcmp(command, "STATUS") == 0) {
    printStatus();
    return;
  }

  if (strcmp(command, "PING") == 0) {
    Serial.println(F("PONG"));
    return;
  }

  Serial.println(F("ERR BAD_COMMAND"));
}

void processManualKey(char key) {
  switch (key) {
    case 'w': driveVelocity( MANUAL_RPM,  MANUAL_RPM); break;
    case 's': driveVelocity(-MANUAL_RPM, -MANUAL_RPM); break;
    case 'a': driveVelocity(-MANUAL_RPM,  MANUAL_RPM); break;
    case 'd': driveVelocity( MANUAL_RPM, -MANUAL_RPM); break;
    case 'x':
      stopMotors();
      Serial.println(F("STOPPED"));
      break;
    case 'p': printStatus(); break;
  }
}

void processSerial() {
  while (Serial.available() > 0) {
    const char input = static_cast<char>(Serial.read());

    // Lower-case keys retain immediate WASD operation without a newline.
    if (lineLength == 0 && strchr("wasdxp", input) != nullptr) {
      processManualKey(input);
      continue;
    }

    if (input == '\r' || input == '\n') {
      if (lineLength > 0) {
        lineBuffer[lineLength] = '\0';
        processLine(lineBuffer);
        lineLength = 0;
      }
      continue;
    }

    if (lineLength < sizeof(lineBuffer) - 1U) {
      lineBuffer[lineLength++] = input;
    } else {
      lineLength = 0;
      Serial.println(F("ERR LINE_TOO_LONG"));
    }
  }
}

void setup() {
  setupPwm20kHz();
  stopMotors();

  pinMode(ENCODER1_A_PIN, INPUT_PULLUP);
  pinMode(ENCODER1_B_PIN, INPUT_PULLUP);
  pinMode(ENCODER2_A_PIN, INPUT_PULLUP);
  pinMode(ENCODER2_B_PIN, INPUT_PULLUP);
  encoderPrevious[0] = static_cast<uint8_t>((digitalRead(ENCODER1_A_PIN) ? 1U : 0U)
                                           | (digitalRead(ENCODER1_B_PIN) ? 2U : 0U));
  encoderPrevious[1] = static_cast<uint8_t>((digitalRead(ENCODER2_A_PIN) ? 1U : 0U)
                                           | (digitalRead(ENCODER2_B_PIN) ? 2U : 0U));
  attachInterrupt(digitalPinToInterrupt(ENCODER1_A_PIN), encoder1AIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER1_B_PIN), encoder1BIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER2_A_PIN), encoder2AIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER2_B_PIN), encoder2BIsr, CHANGE);

  motors[0] = {{0.00060f, 0.00120f, 0.00001f, 0.0f, 0.0f, 0.0f},
               0.0f, 0.0f, 0.0f, ENCODER1_CPR, readEncoder(0)};
  motors[1] = {{0.00060f, 0.00120f, 0.00001f, 0.0f, 0.0f, 0.0f},
               0.0f, 0.0f, 0.0f, ENCODER2_CPR, readEncoder(1)};

  Serial.begin(115200);
  Serial.println(F("READY MEGA_MOTION_V2"));
  Serial.println(F("DRIVE left_rpm right_rpm | MOVE seq left_deg right_deg max_rpm"));
  Serial.println(F("HB | STOP | STATUS | PING"));
  nextControlUs = micros() + CONTROL_PERIOD_US;
  nextPositionUs = micros() + POSITION_PERIOD_US;
  nextTelemetryMs = millis() + TELEMETRY_PERIOD_MS;
  lastDriveCommandMs = millis();
  lastPositionLinkMs = millis();
}

void loop() {
  processSerial();

  const uint32_t nowUs = micros();
  if (static_cast<int32_t>(nowUs - nextControlUs) >= 0) {
    nextControlUs += CONTROL_PERIOD_US;
    controlStep();
  }

  if (controlMode == ControlMode::POSITION
      && static_cast<int32_t>(nowUs - nextPositionUs) >= 0) {
    nextPositionUs += POSITION_PERIOD_US;
    positionStep();
  }

  const uint32_t nowMs = millis();
  if (static_cast<int32_t>(nowMs - nextTelemetryMs) >= 0) {
    nextTelemetryMs += TELEMETRY_PERIOD_MS;
    printStatus();
  }

  if (controlMode == ControlMode::MANUAL) {
    const uint32_t driveSilenceMs = static_cast<uint32_t>(nowMs - lastDriveCommandMs);
    if (driveSilenceMs > DRIVE_TIMEOUT_MS) {
      stopMotors();
      Serial.println(F("ERR DRIVE_TIMEOUT"));
    }
  } else if (controlMode == ControlMode::POSITION) {
    const uint32_t linkSilenceMs = static_cast<uint32_t>(nowMs - lastPositionLinkMs);
    if (linkSilenceMs > LINK_TIMEOUT_MS) {
      const uint32_t stoppedSequence = activeSequence;
      stopMotors();
      Serial.print(F("ERR LINK_TIMEOUT "));
      Serial.println(stoppedSequence);
    }
  }
}