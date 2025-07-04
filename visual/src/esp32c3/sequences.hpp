#ifndef SEQUENCES_HPP
#define SEQUENCES_HPP

#include <Arduino.h> // Include for basic types if needed

// --- Shared Struct Definition ---
// Defined ONCE here for all files that include this header
struct SequenceStep {
  int durationMs;
  // Group A
  int startFreqA;
  int endFreqA;
  int startDutyA;
  int endDutyA;
  int startBrightnessA; // 0-100 (%)
  int endBrightnessA;   // 0-100 (%)
  int waveformA;
  // Group B
  int startFreqB;
  int endFreqB;
  int startDutyB;
  int endDutyB;
  int startBrightnessB; // 0-100 (%)
  int endBrightnessB;   // 0-100 (%)
  int waveformB;
};

// --- Declare Global Variables (defined in main.cpp) ---
// 'extern' tells the compiler these exist elsewhere
extern volatile bool isSequenceRunning;
// extern volatile bool stopRequested; // Add if needed by sequences.cpp directly

// --- Declare Functions Defined Elsewhere ---
// This function is defined in main.cpp but called by sequences in sequences.cpp
bool runSmoothSequence(const SequenceStep* steps, int stepCount);

// --- Declare Sequence Functions ---
// These are functions callable from loop() in main.cpp

// Defined in main.cpp
void rampTestSequence();

// Defined in sequences.cpp
void h_gamma_3();
void three_step_frac();

// Python script will append more declarations here, like:
// void some_new_sequence(); // Generated from some_new_sequence.json


#endif // SEQUENCES_HPP
void _40hzsplit(); // Generated from 40hzsplit.json
