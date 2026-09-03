// =====================================================================
//  agrix_tank_pump.ino  —  물통 수위(초음파) + 펌프 PWM  하위 제어기
//  Arduino Mega 2560 / USB CDC 115200
// ---------------------------------------------------------------------
//  이 스케치는 주행 제어(agrix_motor_mega.ino) 와 **별개**입니다.
//  브링업/모니터링 용도로 따로 올려서 쓰다가, 나중에 한 보드에 합칠 때는
//  아래 명령 2개(PUMP / TANK 텔레메트리)만 모션 펌웨어에 옮기면 됩니다.
//  핀은 모션 펌웨어와 겹치지 않게 골라 두었습니다.
//      모션 사용 핀 : 6, 7, 44, 45 (PWM) / 2, 3, 18, 19 (엔코더 인터럽트)
//
//  ★ 핀맵은 임시값입니다. 실제 배선이 정해지면 아래 상수만 고치세요.
// ---------------------------------------------------------------------
//  프로토콜 (줄 단위 ASCII)
//    Pi -> Mega
//      PUMP <0..255>     펌프 듀티 지정 (0 = 정지)
//      PUMP OFF          펌프 정지
//      STOP              펌프 정지 (PUMP OFF 와 동일)
//      STATUS            텔레메트리 1줄 즉시 전송
//      PING              -> PONG
//      CAL <높이mm> <empty mm> <오프셋mm>   물통 형상 런타임 보정
//
//    Mega -> Pi (기본 5Hz)
//      TANK <dist_mm> <level_mm> <pct> <탱크상태> <duty> <펌프상태> <flags>
//        탱크상태 : FULL | OK | LOW | EMPTY | FAULT
//        펌프상태 : OFF | ON | LOCK        (LOCK = 인터록으로 강제 차단)
//        flags    : 비트 0 = 물부족 래치
//                   비트 1 = 초음파 이상
//                   비트 2 = 명령 워치독 정지
//                   비트 3 = 연속 운전 시간 초과
//      READY AGRIX_TANK_PUMP_V1
//      ERR <사유>
// =====================================================================

// ------------------------- 핀맵 (임시) -------------------------------
constexpr uint8_t US_TRIG_PIN = 30;   // HC-SR04 Trig
constexpr uint8_t US_ECHO_PIN = 31;   // HC-SR04 Echo  (5V 로직, 분압 불필요)
constexpr uint8_t PUMP_PWM_PIN = 9;   // LR7843 MOSFET 게이트 (Timer2 OC2B)

// ------------------------- 물통 형상 ---------------------------------
//  센서는 물통 **뚜껑 안쪽**에서 아래(수면)를 봅니다.
//
//      ┌──────────────┐  ← 초음파 센서 면
//      │   공기       │     dist = 센서~수면
//      │ ─ ─ ─ ─ ─ ─  │  ← 만수위 (센서에서 SENSOR_OFFSET_MM 아래)
//      │   물         │     level = TANK_HEIGHT_MM - (dist - OFFSET)
//      └──────────────┘  ← 바닥
//
//  [주의] HC-SR04 는 약 20mm 이내를 못 읽습니다(사각지대). 만수위에서
//    dist 가 그 근처면 값이 튀므로 센서를 만수위보다 최소 40mm 위에
//    다세요. 그러면 dist 범위가 40mm(만수) ~ 340mm(빈통) 로 안전합니다.
int16_t tankHeightMm = 300;      // 물통 높이(최대 수심) ≈ 30cm
int16_t emptyLevelMm = 50;       // 이 수심 이하 = empty ≈ 5cm
int16_t emptyClearMm = 70;       // 히스테리시스: 이 위로 올라와야 해제
int16_t sensorOffsetMm = 40;     // 센서 면 ~ 만수위 간격

// 상태 구분 문턱 (수심 기준, 퍼센트 아님)
constexpr int16_t LOW_LEVEL_MM = 90;    // 이 아래면 LOW 경고
constexpr int16_t FULL_LEVEL_MM = 270;  // 이 위면 FULL

// ------------------------- 동작 상수 ---------------------------------
constexpr uint32_t PING_INTERVAL_MS = 70;    // 초음파 발사 간격(에코 간섭 방지)
constexpr uint32_t TELEMETRY_MS = 200;       // 5Hz
constexpr uint32_t PUMP_WATCHDOG_MS = 1500;  // 무명령 시 펌프 정지
// 연속 운전 상한(건식/과열 보호). 여기 걸리면 duty 를 0 으로 잠급니다.
//   해제하려면 PUMP 0 (또는 STOP) 을 먼저 보낸 뒤 다시 켜세요.
//   같은 듀티를 계속 재전송하는 것만으로는 풀리지 않습니다(의도된 동작).
constexpr uint32_t PUMP_MAX_RUN_MS = 120000;
constexpr uint8_t MEDIAN_N = 5;              // 중앙값 필터 표본 수
// pulseIn 은 블로킹입니다. 반사가 없으면 이 시간만큼 루프가 멈춥니다.
//   70ms 주기에서 최악 25ms 정지 — 시리얼은 버퍼에 쌓이고 PWM 은 하드웨어라
//   영향 없습니다. 모션 펌웨어와 합칠 때는 이 블로킹을 반드시 없애세요
//   (핀 체인지 인터럽트 + micros() 로 비동기 측정).
constexpr uint32_t ECHO_TIMEOUT_US = 25000;  // ≈ 4.3m
constexpr uint8_t PUMP_RAMP_STEP = 8;        // 틱당 듀티 변화 상한(돌입전류 억제)
constexpr uint32_t RAMP_INTERVAL_MS = 20;

// 초음파를 못 읽을 때 펌프를 돌릴 것인가.
//   false(기본) = 물이 있는지 확인 못 하면 돌리지 않는다(건식 운전 방지).
constexpr bool ALLOW_PUMP_ON_FAULT = false;

// ------------------------- 상태 --------------------------------------
int16_t samples[MEDIAN_N];
uint8_t sampleIndex = 0;
uint8_t sampleCount = 0;

int16_t distMm = -1;
int16_t levelMm = 0;
uint8_t levelPct = 0;
bool emptyLatched = false;
bool sensorFault = true;         // 첫 유효 측정 전까지 이상으로 본다
uint8_t faultStreak = 0;

uint8_t pumpTarget = 0;          // 상위가 요청한 듀티
uint8_t pumpDuty = 0;            // 실제 출력 듀티(램프 적용)
bool pumpLocked = false;
bool watchdogTripped = false;
bool runtimeTripped = false;

uint32_t lastPingMs = 0;
uint32_t lastTelemetryMs = 0;
uint32_t lastCommandMs = 0;
uint32_t lastRampMs = 0;
uint32_t pumpStartedMs = 0;

char lineBuf[64];
uint8_t lineLen = 0;

// =====================================================================
//  초음파
// =====================================================================
int16_t pingOnce() {
  digitalWrite(US_TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(US_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(US_TRIG_PIN, LOW);

  const uint32_t us = pulseIn(US_ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (us == 0) return -1;                       // 타임아웃 = 반사 없음

  // 음속 343 m/s (20도 기준) -> 왕복이므로 절반.
  //   0.343 mm/us / 2 = 0.1715
  const int32_t mm = (int32_t)(us * 0.1715f);
  if (mm < 20 || mm > 4000) return -1;          // 사각지대 / 사거리 밖
  return (int16_t)mm;
}

int16_t medianOfSamples() {
  int16_t sorted[MEDIAN_N];
  uint8_t n = 0;
  for (uint8_t i = 0; i < sampleCount; i++) {
    if (samples[i] > 0) sorted[n++] = samples[i];
  }
  if (n == 0) return -1;
  for (uint8_t i = 1; i < n; i++) {            // 삽입 정렬 (n<=5)
    const int16_t key = sorted[i];
    int8_t j = i - 1;
    while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j--; }
    sorted[j + 1] = key;
  }
  return sorted[n / 2];
}

void updateTank() {
  const int16_t raw = pingOnce();
  samples[sampleIndex] = raw;
  sampleIndex = (sampleIndex + 1) % MEDIAN_N;
  if (sampleCount < MEDIAN_N) sampleCount++;

  const int16_t med = medianOfSamples();
  if (med < 0) {
    if (faultStreak < 250) faultStreak++;
    if (faultStreak >= 5) {                    // 연속 5회 실패해야 이상 확정
      sensorFault = true;
      distMm = -1;
    }
    return;
  }

  faultStreak = 0;
  sensorFault = false;
  distMm = med;

  int32_t level = (int32_t)tankHeightMm - ((int32_t)med - sensorOffsetMm);
  if (level < 0) level = 0;
  if (level > tankHeightMm) level = tankHeightMm;
  levelMm = (int16_t)level;
  levelPct = (uint8_t)((level * 100L) / (tankHeightMm > 0 ? tankHeightMm : 1));

  // 물부족 판정은 래치 + 히스테리시스.
  //   출렁임으로 한 번 튄 값에 펌프가 껐다 켜졌다 하지 않게 한다.
  if (levelMm <= emptyLevelMm) emptyLatched = true;
  else if (levelMm >= emptyClearMm) emptyLatched = false;
}

const __FlashStringHelper *tankStateName() {
  if (sensorFault) return F("FAULT");
  if (emptyLatched) return F("EMPTY");
  if (levelMm <= LOW_LEVEL_MM) return F("LOW");
  if (levelMm >= FULL_LEVEL_MM) return F("FULL");
  return F("OK");
}

// =====================================================================
//  펌프
// =====================================================================
void applyPump() {
  // --- 인터록: 물이 없거나 확인 불가면 무조건 차단 ---
  pumpLocked = emptyLatched || (sensorFault && !ALLOW_PUMP_ON_FAULT);

  const uint32_t now = millis();
  watchdogTripped = (pumpTarget > 0) &&
                    ((uint32_t)(now - lastCommandMs) > PUMP_WATCHDOG_MS);

  if (pumpDuty > 0 && pumpStartedMs != 0 &&
      (uint32_t)(now - pumpStartedMs) > PUMP_MAX_RUN_MS) {
    runtimeTripped = true;
  }

  uint8_t want = pumpTarget;
  if (pumpLocked || watchdogTripped || runtimeTripped) want = 0;

  // 램프: 돌입 전류와 MOSFET 발열을 줄인다. 정지는 즉시.
  if (want == 0) {
    pumpDuty = 0;
  } else if ((uint32_t)(now - lastRampMs) >= RAMP_INTERVAL_MS) {
    lastRampMs = now;
    if (want > pumpDuty) {
      pumpDuty = (want - pumpDuty > PUMP_RAMP_STEP) ? pumpDuty + PUMP_RAMP_STEP
                                                    : want;
    } else if (want < pumpDuty) {
      pumpDuty = (pumpDuty - want > PUMP_RAMP_STEP) ? pumpDuty - PUMP_RAMP_STEP
                                                    : want;
    }
  }

  if (pumpDuty == 0) {
    pumpStartedMs = 0;
    runtimeTripped = runtimeTripped && (pumpTarget > 0);  // 정지 명령 시 해제
  } else if (pumpStartedMs == 0) {
    pumpStartedMs = now;
  }

  analogWrite(PUMP_PWM_PIN, pumpDuty);
}

const __FlashStringHelper *pumpStateName() {
  if (pumpLocked || watchdogTripped || runtimeTripped) return F("LOCK");
  return pumpDuty > 0 ? F("ON") : F("OFF");
}

// =====================================================================
//  텔레메트리 / 명령
// =====================================================================
void sendTelemetry() {
  uint8_t flags = 0;
  if (emptyLatched) flags |= 0x01;
  if (sensorFault) flags |= 0x02;
  if (watchdogTripped) flags |= 0x04;
  if (runtimeTripped) flags |= 0x08;

  Serial.print(F("TANK "));
  Serial.print(distMm);
  Serial.print(' ');
  Serial.print(levelMm);
  Serial.print(' ');
  Serial.print(levelPct);
  Serial.print(' ');
  Serial.print(tankStateName());
  Serial.print(' ');
  Serial.print(pumpDuty);
  Serial.print(' ');
  Serial.print(pumpStateName());
  Serial.print(' ');
  Serial.println(flags);
}

void handleLine(char *line) {
  while (*line == ' ') line++;
  if (*line == '\0') return;

  char *cmd = strtok(line, " ");
  if (cmd == NULL) return;

  if (strcmp(cmd, "PING") == 0) {
    Serial.println(F("PONG"));
    return;
  }
  if (strcmp(cmd, "STATUS") == 0) {
    sendTelemetry();
    return;
  }
  if (strcmp(cmd, "STOP") == 0) {
    pumpTarget = 0;
    runtimeTripped = false;
    lastCommandMs = millis();
    return;
  }
  if (strcmp(cmd, "PUMP") == 0) {
    char *arg = strtok(NULL, " ");
    if (arg == NULL) { Serial.println(F("ERR PUMP_NO_ARG")); return; }
    if (strcmp(arg, "OFF") == 0 || strcmp(arg, "off") == 0) {
      pumpTarget = 0;
      runtimeTripped = false;
    } else {
      const long v = atol(arg);
      if (v < 0 || v > 255) { Serial.println(F("ERR PUMP_RANGE")); return; }
      if (v > 0 && pumpTarget == 0) runtimeTripped = false;  // 새로 켜는 명령
      pumpTarget = (uint8_t)v;
    }
    lastCommandMs = millis();
    return;
  }
  if (strcmp(cmd, "CAL") == 0) {
    char *a = strtok(NULL, " ");
    char *b = strtok(NULL, " ");
    char *c = strtok(NULL, " ");
    if (a == NULL || b == NULL || c == NULL) {
      Serial.println(F("ERR CAL_ARGS"));
      return;
    }
    tankHeightMm = (int16_t)atol(a);
    emptyLevelMm = (int16_t)atol(b);
    sensorOffsetMm = (int16_t)atol(c);
    emptyClearMm = emptyLevelMm + 20;
    Serial.print(F("CAL OK "));
    Serial.print(tankHeightMm); Serial.print(' ');
    Serial.print(emptyLevelMm); Serial.print(' ');
    Serial.println(sensorOffsetMm);
    return;
  }
  Serial.println(F("ERR BAD_COMMAND"));
}

void readSerial() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      handleLine(lineBuf);
      lineLen = 0;
      continue;
    }
    if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
      Serial.println(F("ERR LINE_TOO_LONG"));
    }
  }
}

// =====================================================================
void setup() {
  pinMode(US_TRIG_PIN, OUTPUT);
  pinMode(US_ECHO_PIN, INPUT);
  pinMode(PUMP_PWM_PIN, OUTPUT);
  digitalWrite(US_TRIG_PIN, LOW);
  analogWrite(PUMP_PWM_PIN, 0);

  for (uint8_t i = 0; i < MEDIAN_N; i++) samples[i] = -1;

  Serial.begin(115200);
  while (!Serial) { ; }
  Serial.println(F("READY AGRIX_TANK_PUMP_V1"));
  Serial.println(F("PUMP <0-255|OFF> | STOP | STATUS | PING | CAL h empty offset"));

  lastCommandMs = millis();
}

void loop() {
  readSerial();

  const uint32_t now = millis();
  if ((uint32_t)(now - lastPingMs) >= PING_INTERVAL_MS) {
    lastPingMs = now;
    updateTank();
  }

  applyPump();

  if ((uint32_t)(now - lastTelemetryMs) >= TELEMETRY_MS) {
    lastTelemetryMs = now;
    sendTelemetry();
  }
}
