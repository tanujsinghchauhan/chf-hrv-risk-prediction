
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

MAX30105 sensor;

// Keep the pins from your existing working sketch.
const int SDA_PIN = 8;
const int SCL_PIN = 9;

// DEMO mode creates simulated metrics for UI demonstration.
// Do NOT use simulated metrics for medical decisions or model training.
const bool DEMO_MODE = true;

const uint32_t FINGER_THRESHOLD = 10000;

unsigned long lastPrint = 0;
unsigned long lastBeat = 0;
unsigned long lastDemoUpdate = 0;
unsigned long demoBeatTime = 0;

float realBPM = 0;
float realRR = 0;
float demoBPM = 74.0;
float demoRR = 810.8;

bool realBeatDetected = false;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);

  Serial.println("ESP32-S3 MAX30102 Monitor");

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: Sensor not detected");
    Serial.println("Check power, GND, SDA, SCL.");
    while (true) delay(1000);
  }

  sensor.setup(0x1F, 4, 2, 100, 411, 4096);
  sensor.setPulseAmplitudeRed(0x10);
  sensor.setPulseAmplitudeIR(0x10);
  sensor.setPulseAmplitudeGreen(0);

  Serial.println("Sensor ready");
  Serial.println(
    "time_ms,IR,RED,finger,beat,BPM,RR_ms,source"
  );
}

void loop() {
  sensor.check();

  while (sensor.available()) {
    uint32_t ir = sensor.getIR();
    uint32_t red = sensor.getRed();
    sensor.nextSample();

    unsigned long now = millis();
    bool finger = ir > FINGER_THRESHOLD;
    bool beat = false;

    if (!finger) {
      realBPM = 0;
      realRR = 0;
      lastBeat = 0;
      demoBPM = 0;
      demoRR = 0;
      demoBeatTime = 0;

      if (now - lastPrint >= 100) {
        Serial.printf(
          "%lu,%lu,%lu,0,0,0.0,0.0,NONE\n",
          now, (unsigned long)ir, (unsigned long)red
        );
        lastPrint = now;
      }
      continue;
    }

    // Attempt genuine pulse detection.
    if (checkForBeat(ir)) {
      if (lastBeat != 0) {
        unsigned long interval = now - lastBeat;

        if (interval >= 300 && interval <= 2000) {
          realRR = interval;
          realBPM = 60000.0 / realRR;
          realBeatDetected = true;
          beat = true;
        }
      }
      lastBeat = now;
    }

    // Explicitly simulated metrics for dashboard layout testing.
    // These are NOT measured physiological values.
    if (DEMO_MODE && !realBeatDetected) {
      if (demoBeatTime == 0) {
        demoBeatTime = now;
        lastDemoUpdate = now;
      }

      if (now - demoBeatTime >= (unsigned long)demoRR) {
        demoBeatTime = now;
        beat = true;
      }

      // Slowly vary the demo values to animate the panel.
      if (now - lastDemoUpdate >= 5000) {
        demoBPM += (demoBPM >= 76.0) ? -1.0 : 1.0;
        demoRR = 60000.0 / demoBPM;
        lastDemoUpdate = now;
      }
    }

    if (now - lastPrint >= 100) {
      if (realBeatDetected) {
        Serial.printf(
          "%lu,%lu,%lu,1,%d,%.1f,%.1f,REAL\n",
          now, (unsigned long)ir, (unsigned long)red,
          beat ? 1 : 0, realBPM, realRR
        );
      } else if (DEMO_MODE) {
        Serial.printf(
          "%lu,%lu,%lu,1,%d,%.1f,%.1f,DEMO_SIMULATED\n",
          now, (unsigned long)ir, (unsigned long)red,
          beat ? 1 : 0, demoBPM, demoRR
        );
      } else {
        Serial.printf(
          "%lu,%lu,%lu,1,0,0.0,0.0,WAITING_FOR_BEAT\n",
          now, (unsigned long)ir, (unsigned long)red
        );
      }

      lastPrint = now;
    }
  }
}
