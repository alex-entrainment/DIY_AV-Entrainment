/***************************************************************************
  ESP32 LED Oscillator System using Native LEDC PWM (Final Version)

  - Controls 6 LED channels using the ESP32's LEDC peripheral.
  - Reads sequence steps defined in SequenceStep struct (via sequences.hpp).
  - Supports smooth parameter transitions (Frequency, Duty, Brightness).
  - Uses a non-blocking timer loop for precise update rates (~1kHz target).
  - Handles RUN/STOP commands via Serial.
  - Assumes external LED driving circuitry (e.g., MOSFETs) connected
    to the specified GPIO pins.
***************************************************************************/

#include <Arduino.h>
#include <cmath>        // Required for M_PI, sinf, fmod
#include "sequences.hpp" // Include SequenceStep struct, sequence declarations, extern vars

// --- LEDC Configuration ---
const int LEDC_FREQ = 5000; // PWM frequency in Hz (5kHz is good for LEDs)
const int LEDC_RESOLUTION_BITS = 12; // Use 12-bit resolution (0-4095 duty cycle)
// Calculate max duty value based on resolution
const uint32_t LEDC_MAX_DUTY = (1 << LEDC_RESOLUTION_BITS) - 1; // = 4095 for 12-bit

// --- Pin Mapping & Channel Count ---
const int NUM_OSCILLATORS = 6; // Number of channels/oscillators to control

// !! EDIT THIS ARRAY TO MATCH YOUR ACTUAL GPIO WIRING !!
// Maps logical channels (0-5) to physical ESP32 GPIO pins.
// Defaulting to left-side header pins (based on image): 6, 7, 8, 9, 10, 20
const int ledcPinMap[NUM_OSCILLATORS] = {
    6,  // Logical Channel 0 -> GPIO 6
    7,  // Logical Channel 1 -> GPIO 7
    8,  // Logical Channel 2 -> GPIO 8 (Also onboard LED)
    9,  // Logical Channel 3 -> GPIO 9
   10,  // Logical Channel 4 -> GPIO 10
   20   // Logical Channel 5 -> GPIO 20
};

// --- Oscillator Class ---
class Oscillator {
public:
  int   freq;       // 0.01 Hz increments
  int   waveform;   // 1 => square, 2 => sine
  int   duty;       // 0..100
  float phase;      // 0..2π
  float brightness; // 0.0 .. 1.0

  Oscillator(int f=100, int wave=1, int d=50, float bright=1.0f)
    : freq(f), waveform(wave), duty(d), phase(0.0f), brightness(bright)
  {}
};

// --- Global Oscillators & Groups ---
// These store the run-time state of the oscillators
static Oscillator oscillators[NUM_OSCILLATORS] = {
  /* channel 0 */ Oscillator(  50, 2, 50, 1.0f), // Initial state placeholders
  /* channel 1 */ Oscillator( 100, 1, 50, 1.0f),
  /* channel 2 */ Oscillator( 200, 2, 50, 1.0f),
  /* channel 3 */ Oscillator( 100, 1, 50, 1.0f),
  /* channel 4 */ Oscillator( 100, 2, 50, 1.0f),
  /* channel 5 */ Oscillator( 100, 1, 50, 1.0f)
};

static const int groupA[3] = {0, 2, 4}; // Logical channels for Group A
static const int groupB[3] = {1, 3, 5}; // Logical channels for Group B

// --- Global State Variables (Defined Here) ---
// Declared 'extern' in sequences.hpp
volatile bool isSequenceRunning = false;
volatile bool stopRequested = false;

// --- LEDC Setup Function ---
// Configures the ESP32 LEDC peripheral for the 6 channels
void setupLEDC() {
    Serial.println("Setting up LEDC PWM channels...");
    bool setup_ok = true;
    for (int ch = 0; ch < NUM_OSCILLATORS; ++ch) {
        // Setup LEDC channel (channel number 0-5, frequency, resolution)
        // ESP32-C3: 6 channels, 4 timers. Assign channels to timers (e.g., ch % 4)
        if (ledcSetup(ch, LEDC_FREQ, LEDC_RESOLUTION_BITS) == 0) { // Returns 0 on failure
             Serial.printf("ERROR: LEDC channel %d setup failed!\n", ch);
             setup_ok = false;
        } else {
             // Attach the GPIO pin to the configured LEDC channel
             ledcAttachPin(ledcPinMap[ch], ch);
             Serial.printf("LEDC channel %d attached to GPIO %d\n", ch, ledcPinMap[ch]);
             ledcWrite(ch, 0); // Ensure LED starts off
        }
    }
    if (!setup_ok) {
        Serial.println("FATAL: One or more LEDC channels failed setup. Halting.");
        while(true) { delay(1000); } // Halt on critical error
    }
    Serial.println("LEDC setup complete.");
}

// --- Helper Functions ---

// Turn off all LEDs using LEDC
void turnOffAllLeds() {
    Serial.println("Turning off all LEDs via LEDC.");
    for (int ch = 0; ch < NUM_OSCILLATORS; ++ch) {
        // Assumes channels 0-5 have been successfully set up
        ledcWrite(ch, 0); // Set duty cycle to 0
    }
}

// Update oscillator phases based on elapsed time (dtMillis)
static void updateOscillatorPhases(uint32_t dtMillis) {
  if (dtMillis == 0) return;
  float dtSec = (float)dtMillis / 1000.0f;
  for (int i = 0; i < NUM_OSCILLATORS; i++) {
    float freqHz = (float)oscillators[i].freq / 100.0f;
    oscillators[i].phase += 2.0f * (float)M_PI * freqHz * dtSec;
    oscillators[i].phase = fmod(oscillators[i].phase, 2.0f * (float)M_PI);
    if (oscillators[i].phase < 0) { oscillators[i].phase += 2.0f * (float)M_PI; }
  }
}

// Get oscillator value [0.0..1.0] scaled by brightness
static float getOscillatorValue(const Oscillator &osc) {
  float baseValue;
  if (osc.waveform == 1) { // Square wave
    float dutyRad = (osc.duty / 100.0f) * 2.0f * (float)M_PI;
    baseValue = (osc.phase < dutyRad) ? 1.0f : 0.0f;
  } else { // Sine wave (default)
    float s = sinf(osc.phase);
    baseValue = (s + 1.0f) * 0.5f;
  }
  return baseValue * osc.brightness;
}

// Linear interpolation for integers
static int linearInterpInt(int startVal, int endVal, float fraction) {
    float diff = (float)(endVal - startVal);
    return startVal + (int)(diff * fraction + 0.5f); // Round
}

// Linear interpolation for floats
static float linearInterpFloat(float startVal, float endVal, float fraction) {
    return startVal + (endVal - startVal) * fraction;
}

// Apply interpolated parameters to an oscillator group
static void applyGroupSmooth(const int* channels, int numCh,
                             int startFreq, int endFreq,
                             int startDuty, int endDuty,
                             int startBrightness, int endBrightness,
                             int waveform,
                             float fraction)
{
  for (int i = 0; i < numCh; i++) {
    int ch_index = channels[i]; // Get logical channel index (0-5)
    if (ch_index < 0 || ch_index >= NUM_OSCILLATORS) continue; // Bounds check

    Oscillator &osc = oscillators[ch_index]; // Access the correct oscillator state
    osc.freq = linearInterpInt(startFreq, endFreq, fraction);
    osc.duty = linearInterpInt(startDuty, endDuty, fraction);
    int currentBrightnessInt = linearInterpInt(startBrightness, endBrightness, fraction);
    osc.brightness = constrain((float)currentBrightnessInt / 100.0f, 0.0f, 1.0f);
    osc.waveform = waveform;
  }
}

// --- Smooth Sequence Runner ---
// Uses non-blocking timer and direct LEDC writes
// Declaration must be in sequences.hpp: bool runSmoothSequence(const SequenceStep* steps, int stepCount);
bool runSmoothSequence(const SequenceStep* steps, int stepCount) {

  // Target loop update interval in microseconds (1000us = 1ms = 1000 Hz)
  const uint32_t LOOP_INTERVAL_MICROS = 1000;

  uint16_t pwmValues[NUM_OSCILLATORS]; // Buffer for calculated PWM duty cycles
  uint32_t loopUpdateCount = 0; // Track number of actual PWM updates per step
  uint32_t lastLoopActivityTime = millis(); // For stuck loop detection

  stopRequested = false; // Ensure stop flag is clear initially
  Serial.println("Sequence starting (using LEDC)...");

  for (int i = 0; i < stepCount; i++) { // Loop through each step in the sequence
    // --- Start of Step Initialization ---
    if (stopRequested) { Serial.println("Stop requested before step start."); turnOffAllLeds(); return false; }

    Serial.print("Starting Step "); Serial.print(i);
    Serial.print(" Duration: "); Serial.print(steps[i].durationMs); Serial.println(" ms");
    // (Add more Serial.print details about Freq/Brightness if desired)

    uint32_t stepStartMillis = millis();
    uint32_t lastLoopStartMicros = micros(); // Timer for this step's update loop
    loopUpdateCount = 0; // Reset diagnostic counter for this step

    // --- Step Execution Loop (Runs until duration expires or stopped) ---
    while (true) {
      uint32_t nowMicros = micros();
      uint32_t nowMillis = millis();

      // --- Check for STOP command frequently ---
      if (Serial.available() > 0) {
          String cmd = Serial.readStringUntil('\n'); cmd.trim();
          if (cmd == "STOP") { Serial.println("STOP received."); stopRequested = true; }
          // Clear any other serial input received while busy
          while(Serial.available() > 0) Serial.read();
      }
      if (stopRequested) { Serial.println("Sequence stopped by command."); turnOffAllLeds(); return false; }
      // --- End STOP Check ---

      // --- Check if step duration is complete ---
      uint32_t elapsedStepMillis = nowMillis - stepStartMillis;
      if (elapsedStepMillis >= steps[i].durationMs) {
          break; // Exit while loop, step finished
      }

      // --- Timed Update Section (Runs approx every LOOP_INTERVAL_MICROS) ---
      uint32_t elapsedMicros = (nowMicros >= lastLoopStartMicros) ? (nowMicros - lastLoopStartMicros) : (UINT32_MAX - lastLoopStartMicros + nowMicros + 1);
      if (elapsedMicros >= LOOP_INTERVAL_MICROS) {

          // --- Calculate Next Interval Start Time ---
          // Correct for drift by adding interval to last start, not using current time
          lastLoopStartMicros += LOOP_INTERVAL_MICROS;
          // Resync if we've fallen significantly behind (e.g., due to long interrupt)
          if ( (nowMicros >= lastLoopStartMicros) ? (nowMicros - lastLoopStartMicros) > LOOP_INTERVAL_MICROS : true) {
               // If predicted next start is still in the past, snap to now
               lastLoopStartMicros = nowMicros;
          }
          // --- End Interval Calculation ---

          lastLoopActivityTime = nowMillis; // Record that we performed an update cycle

          // Calculate progress fraction (0.0 to 1.0) for this step
          float frac = (float)elapsedStepMillis / (float)steps[i].durationMs;
          frac = constrain(frac, 0.0f, 1.0f);

          // Update oscillator parameters (freq, duty, brightness, waveform) based on fraction
          applyGroupSmooth(groupA, 3, /* Group A params */ steps[i].startFreqA, steps[i].endFreqA, steps[i].startDutyA, steps[i].endDutyA, steps[i].startBrightnessA, steps[i].endBrightnessA, steps[i].waveformA, frac);
          applyGroupSmooth(groupB, 3, /* Group B params */ steps[i].startFreqB, steps[i].endFreqB, steps[i].startDutyB, steps[i].endDutyB, steps[i].startBrightnessB, steps[i].endBrightnessB, steps[i].waveformB, frac);

          // Update oscillator phases based on the fixed loop interval time
          updateOscillatorPhases(LOOP_INTERVAL_MICROS / 1000); // dt must be in ms

          // Calculate all 6 PWM duty cycle values based on current oscillator state
          for (int ch = 0; ch < NUM_OSCILLATORS; ch++) {
            pwmValues[ch] = (uint16_t)(getOscillatorValue(oscillators[ch]) * LEDC_MAX_DUTY); // Scale to LEDC resolution (0-4095)
          }

          // *** Update all 6 LEDC Channels ***
          for (int ch = 0; ch < NUM_OSCILLATORS; ++ch) {
              // Constrain just in case calculation slightly exceeds max duty
              uint32_t dutyCycle = constrain(pwmValues[ch], 0, LEDC_MAX_DUTY);
              ledcWrite(ch, dutyCycle); // Write new duty cycle to LEDC hardware channel
          }
          // ********************************

          loopUpdateCount++; // Increment update counter for diagnostics
      } else {
          // Interval hasn't passed, yield CPU briefly to allow background tasks
          yield(); // or delay(0); Can be important for stability
      }

      // --- Failsafe Check ---
      // If no update cycle has happened for several seconds, something is wrong
      if (nowMillis - lastLoopActivityTime > 5000) { // 5 second timeout
           Serial.println("ERROR: Main update loop seems stuck! Breaking step.");
           // Consider adding more recovery or error reporting here
           return false; // Indicate sequence failure
      }

    } // End while(true) step execution loop

    // --- End of Step Cleanup/Final State ---
    Serial.print("Step "); Serial.print(i); Serial.print(" finished. Total Updates: "); Serial.println(loopUpdateCount);

    // Ensure parameters are set to their final step values (fraction = 1.0)
    applyGroupSmooth(groupA, 3, steps[i].startFreqA, steps[i].endFreqA, steps[i].startDutyA, steps[i].endDutyA, steps[i].startBrightnessA, steps[i].endBrightnessA, steps[i].waveformA, 1.0f);
    applyGroupSmooth(groupB, 3, steps[i].startFreqB, steps[i].endFreqB, steps[i].startDutyB, steps[i].endDutyB, steps[i].startBrightnessB, steps[i].endBrightnessB, steps[i].waveformB, 1.0f);

    // Perform one last phase update based on time since last actual update
    uint32_t finalDtMillis = (millis() > lastLoopActivityTime) ? (millis() - lastLoopActivityTime) : 0;
    updateOscillatorPhases(finalDtMillis);

    // Calculate and write the final PWM values
    for (int ch = 0; ch < NUM_OSCILLATORS; ch++) {
        pwmValues[ch] = (uint16_t)(getOscillatorValue(oscillators[ch]) * LEDC_MAX_DUTY);
        uint32_t dutyCycle = constrain(pwmValues[ch], 0, LEDC_MAX_DUTY);
        ledcWrite(ch, dutyCycle);
    }
    // --- End Final State ---

  } // End of for loop iterating through steps

  Serial.println("Sequence completed normally!");
  // turnOffAllLeds(); // Optional: Turn off LEDs at the very end
  return true; // Indicate normal completion
}


// --- Example Sequence (Defined in this file if needed) ---
// Declaration should be in sequences.hpp: void rampTestSequence();
void rampTestSequence() {
  // This sequence function calls the main runSmoothSequence executor
  SequenceStep steps[] = {
    { .durationMs = 30000, .startFreqA = 1, .endFreqA = 4000, .startDutyA = 50, .endDutyA = 50, .startBrightnessA = 50, .endBrightnessA = 100, .waveformA = 2, /* Sine */
      .startFreqB = 1, .endFreqB = 4000, .startDutyB = 50, .endDutyB = 50, .startBrightnessB = 50, .endBrightnessB = 100, .waveformB = 1 /* Square */ },
    { .durationMs = 30000, .startFreqA = 4000, .endFreqA = 4000, .startDutyA = 50, .endDutyA = 50, .startBrightnessA = 100, .endBrightnessA = 20, .waveformA = 2,
      .startFreqB = 4000, .endFreqB = 4000, .startDutyB = 50, .endDutyB = 50, .startBrightnessB = 100, .endBrightnessB = 20, .waveformB = 1 }
  };
  // Assume isSequenceRunning and runSmoothSequence are accessible (declared extern in .hpp)
  isSequenceRunning = true;
  runSmoothSequence(steps, sizeof(steps) / sizeof(steps[0]));
  isSequenceRunning = false;
}


// --- Arduino Setup ---
void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for serial connection
  delay(500);
  Serial.println("\n--- ESP32 LED Oscillator Controller (LEDC Version) ---");

  // Initialize LEDC PWM system
  setupLEDC(); // Replaces I2C/PCA9685 init

  Serial.println("Setup complete. Ready for commands.");
  Serial.println("Available commands: RUN:<name>, STOP");
  // This line below is managed by the Python converter script
  Serial.println("Known sequence names: rampTestSequence, h_gamma_3, three_step_frac, _40hzsplit");
  Serial.println("---------------------------------------");
}

// --- Arduino Loop ---
void loop() {
  // Only process commands if no sequence is currently running
  if (!isSequenceRunning && Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n'); cmd.trim();

    if (cmd.length() > 0) { // Process only non-empty commands
         Serial.print("Received command: "); Serial.println(cmd);

         if (cmd.startsWith("RUN:")) {
           String seqName = cmd.substring(4);
           Serial.print("Attempting to run sequence: "); Serial.println(seqName);

           // --- Sequence Dispatch (Calls functions declared in sequences.hpp) ---
           // The Python script adds more 'else if' blocks here automatically
           if (seqName == "rampTestSequence") { rampTestSequence(); }
           else if (seqName == "h_gamma_3") { h_gamma_3(); }
           else if (seqName == "three_step_frac") { three_step_frac(); }
           else if (seqName == "_40hzsplit") { _40hzsplit(); }
           // Add other 'else if' blocks for sequences defined in sequences.cpp
           else { // This else block MUST be last
             Serial.println("Error: Unknown sequence name.");
           }
           // -----------------------------------------------------------

           Serial.println("Sequence finished or stopped. Ready for new commands.");
           Serial.println("---------------------------------------");

         } else if (cmd == "STOP") {
           Serial.println("Command STOP ignored (no sequence running).");
         } else { // Unknown command format
           Serial.println("Error: Unknown command.");
         }
    }
     // Optional: Consume any remaining serial data after processing a command
     // while(Serial.available() > 0) { Serial.read(); }
  }

  // No delay() needed in loop() itself. Either the sequence runs (blocking this loop),
  // or this loop runs quickly checking Serial. The yield() inside runSmoothSequence
  // helps prevent starving background tasks when a sequence *is* running.
}