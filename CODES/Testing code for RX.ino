// THIS CODE IS NOT DEVELOPED BY ME AND HAS BEEN COPIED FROM SOME WEBSITES. 
// All credits to them for developing this code

#include <IBusBM.h>  // Library to read iBUS signals

#define IBUS_RX_PIN 2  // iBUS input from FS-iA6
#define PPM_OUT_PIN 9  // PPM output to Pixhawk

IBusBM ibus;  // iBUS object

// PPM configuration
#define CHANNELS 8  // Number of channels in PPM
#define PPM_FRAME_LENGTH 22500  // Total frame length in microseconds
#define PPM_PULSE_LENGTH 400  // Length of the PPM sync pulse

int ppmValues[CHANNELS] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500};  // Default neutral values

void setup() {
    Serial.begin(115200);  // Debugging if needed
    ibus.begin(IBUS_RX_PIN);  // Start iBUS communication
    
    pinMode(PPM_OUT_PIN, OUTPUT);
    digitalWrite(PPM_OUT_PIN, LOW);

    // Start generating PPM signal
    startPPM();
}

// Function to generate PPM signal using timer interrupt
void startPPM() {
    cli();  // Disable interrupts
    TCCR1A = 0;  // Clear Timer1 registers
    TCCR1B = 0;
    
    OCR1A = PPM_FRAME_LENGTH / 2;  // Set the compare match register
    TCCR1B |= (1 << WGM12);  // CTC mode
    TCCR1B |= (1 << CS11);  // Prescaler 8
    TIMSK1 |= (1 << OCIE1A);  // Enable compare match interrupt

    sei();  // Enable interrupts
}

// Timer1 interrupt to generate PPM signal
ISR(TIMER1_COMPA_vect) {
    static byte channel = 0;
    static unsigned int pulse = 0;
    
    if (channel < CHANNELS) {
        pulse = ppmValues[channel];
        channel++;
    } else {
        pulse = PPM_FRAME_LENGTH - (CHANNELS * 2000);  // End of frame sync
        channel = 0;
    }

    digitalWrite(PPM_OUT_PIN, HIGH);
    delayMicroseconds(PPM_PULSE_LENGTH);
    digitalWrite(PPM_OUT_PIN, LOW);
    
    OCR1A = pulse;
}

// Function to update PPM values from iBUS
void loop() {
    if (ibus.read()) {  // If new data is available
        for (int i = 0; i < CHANNELS; i++) {
            ppmValues[i] = ibus.getChannel(i);
        }
    }
}
