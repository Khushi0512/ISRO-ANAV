#include <IBusBM.h>

#define IBUS_RX_PIN 0   // iBUS input from FS-i6 receiver (Connect to RX pin of Arduino)
#define PPM_OUT_PIN 9   // PPM output to Pixhawk

IBusBM ibus;

const int numChannels = 6;
int ppmValues[numChannels];

void setup() {
    Serial.begin(115200);   // Use Serial for iBUS (Must connect FS-i6 to RX)
    ibus.begin(Serial);     // Use Hardware Serial for iBUS

    pinMode(PPM_OUT_PIN, OUTPUT);
    Serial.println("iBUS to PPM converter started");
}

void loop() {
    for (int i = 0; i < numChannels; i++) {
        ppmValues[i] = ibus.readChannel(i);  // Directly read channel values
    }

    sendPPM();
}

void sendPPM() {
    digitalWrite(PPM_OUT_PIN, HIGH);
    delayMicroseconds(300);  // Sync pulse

    for (int i = 0; i < numChannels; i++) {
        digitalWrite(PPM_OUT_PIN, LOW);
        delayMicroseconds(500);  // Channel separation
        digitalWrite(PPM_OUT_PIN, HIGH);
        delayMicroseconds(map(ppmValues[i], 1000, 2000, 1000, 2000));  // Pulse width
    }

    digitalWrite(PPM_OUT_PIN, LOW);
}
