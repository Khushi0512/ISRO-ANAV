#include <IBusBM.h>  // Ensure this library is installed

#define IBUS_RX_PIN 0   // iBUS input (Serial RX)
#define PPM_OUT_PIN 9   // PPM output to Pixhawk

IBusBM ibus;
const int numChannels = 6;  // Adjust if needed
int ppmValues[numChannels];

void setup() {
    Serial.begin(115200);  // iBUS input (disconnect USB when running)
    ibus.begin(Serial);    // Initialize iBUS

    pinMode(PPM_OUT_PIN, OUTPUT);
    Serial.println("✅ iBUS to PPM Converter Started");
}

void loop() {
    for (int i = 0; i < numChannels; i++) {
        ppmValues[i] = ibus.readChannel(i);
    }
    sendPPM();
}

void sendPPM() {
    digitalWrite(PPM_OUT_PIN, HIGH);
    delayMicroseconds(300);

    for (int i = 0; i < numChannels; i++) {
        digitalWrite(PPM_OUT_PIN, LOW);
        delayMicroseconds(500);
        digitalWrite(PPM_OUT_PIN, HIGH);
        delayMicroseconds(map(ppmValues[i], 1000, 2000, 1000, 2000));
    }
    digitalWrite(PPM_OUT_PIN, LOW);
}
