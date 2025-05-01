# Visual

## Hardware Components

> *(Ensure component choices are compatible with 3.3V logic level output from ESP32)*  

1. **Microcontroller:**
    * [ESP32-C3 SuperMini](https://www.amazon.com/ESP32-C3-SuperMini-Development-Board-Processor/dp/B0B512W55Q) (or similar ESP32-C3 board)
    * Compatible Expansion/Breakout Board (like the one pictured) for easy access to pins.
2. **LED Driving Circuitry (Essential - Build Separately):**
    * **6x Logic-Level N-Channel MOSFETs:** E.g., IRLB8721, IRLZ44N (verify gate threshold voltage is suitable for 3.3V drive), or similar. Choose based on LED current requirements.
    * **6x Gate Resistors:** ~100-220Ω (connects ESP32 GPIO to MOSFET Gate). Protects GPIO.
    * **6x Gate Pull-Down Resistors:** ~10kΩ (connects MOSFET Gate to Ground). Ensures MOSFET stays off when ESP32 pin is floating (e.g., during boot).
3. **LEDs:**
    * 3x High-power (~0.3A+) cool white LEDs
    * 3x High-power (~0.3A+) warm white LEDs (Adjust types/colors as desired)
4. **LED Current Limiting Resistors:**
    * **6x Power Resistors:** Value depends on your specific LEDs (forward voltage, current rating) and the voltage of your separate LED power supply (e.g., 5V). Must be rated for the power they will dissipate (P = I²R). Example: For a 3.3V Vf LED at 300mA from a 5V supply, R = (5V - 3.3V) / 0.3A = 1.7V / 0.3A ≈ 5.6Ω. Power = 0.3A * 1.7V = 0.51W (Use a 1W or 2W resistor). **Calculate carefully for your specific components!**
5. **Power Supply:**
    * Separate Power Supply for LEDs (e.g., 5V, 12V) capable of handling the total current of all LEDs at max brightness (e.g., 6 * 0.3A = 1.8A minimum, recommend 2A+).
    * Power Supply for ESP32: Via USB-C port on the SuperMini.
6. **Wiring:** Jumper wires, hookup wire (appropriate gauge for LED current).
7. **Device Stand/Enclosure:** Optional, as needed.

## Connections (Conceptual)

* **ESP32 -> MOSFETs:** Choose 6 GPIO pins on the ESP32 (defined in `main.cpp`'s `ledcPinMap`). Connect each chosen GPIO pin through a ~220Ω resistor to the Gate pin of a corresponding MOSFET.
* **MOSFET Gates -> GND:** Connect each MOSFET Gate pin through a ~10kΩ resistor to the common Ground.
* **MOSFET Sources -> GND:** Connect all MOSFET Source pins to the common Ground.
* **LED Power Supply -> LEDs -> Resistors -> MOSFETs:** Connect the positive terminal of the LED power supply to the anode (+) of each LED (or LED string). Connect the cathode (-) of each LED to one end of its calculated power resistor. Connect the other end of the power resistor to the Drain pin of the corresponding MOSFET.
* **Power & Ground:**
  * Power the ESP32 via its USB-C port.
    > **Crucially, connect the Ground (GND) of the ESP32 to the Ground of the MOSFET circuit and the Ground of the LED power supply.**  
    *All grounds must be common.*

> **Do NOT connect the LEDs directly to the ESP32 GPIO pins.**

![device_front](https://github.com/user-attachments/assets/afbdc4b5-5a0f-4d13-8dac-c4a8a1637cc8)

![device_top](https://github.com/user-attachments/assets/61fd767d-c33e-4d92-b580-0ad7b5000d24)

![device_front](https://github.com/user-attachments/assets/6845f83e-67bc-4779-a394-87a103a8d458)

---

## Tools Required

1. **Soldering iron + solder + flux** (for assembling driver circuits)
2. **Wire strippers/cutters**
3. **Multimeter** (highly recommended for verifying connections, voltages, resistances)
4. **Breadboard** (optional for prototyping driver circuit)
5. **Computer (Windows Recommended for Dev):** For running GUI, converter script, PlatformIO.
6. **Computer (Raspberry Pi Optional for Mobile):** For running `controller.py`.
7. **USB-C Cable** (for ESP32 power and programming/serial)

## Circuit Diagram

![image](https://github.com/user-attachments/assets/53827ceb-c08c-43f5-a1c4-34f3ca5408e9)

---

## System Architecture

The system now consists of two main parts: the ESP32 firmware and the host control software.

1. **ESP32 Firmware (C++ / PlatformIO):**
    * Runs directly on the ESP32-C3 SuperMini.
    * **`main.cpp`**: Initializes hardware (LEDC PWM), handles USB Serial communication (receiving `RUN:`/`STOP:` commands), manages the main state (`isSequenceRunning`), calls sequence functions, and contains core helper functions (`runSmoothSequence`, `applyGroupSmooth`, etc.).
    * **`sequences.hpp`**: Header file defining the `SequenceStep` struct and declaring all available sequence functions and necessary shared variables/functions (`extern`).
    * **`sequences.cpp`**: Source file containing the implementations of specific light sequences (e.g., `h_gamma_3`, `rampTestSequence`, and functions auto-generated from JSON files).
    * Uses the ESP32's native **LEDC peripheral** for precise PWM generation on 6 GPIO pins.

2. **Host Software (Python - Runs on PC or Raspberry Pi):**
    * **`sequence_editor.py` (PyQt5 GUI)**: (Runs on Dev PC) Visual tool to design multi-step sequences, configure oscillators, brightness, and audio parameters. Saves sequences to `.json` files. Can optionally generate corresponding `.wav` audio files using the `sound_creator.py` engine.
    * **`sequence_model.py`**: (Used by GUI) Data classes defining the structure of sequences stored in JSON.
    * **`setup.py`**: (Run once per host machine) Configures environment-specific settings (serial port, paths for converter) and saves them to `config.ini`.
    * **`config.ini`**: Stores configuration settings read by the other Python scripts.
    * **`sound_creator.py`**: (Used by GUI) The audio generation engine. Contains various synthesis functions and helpers to create complex audio waveforms based on parameters defined in the GUI. See the [Audio README](./README_Audio.md) for detailed information on its functionality and usage.
    * **`json_to_cpp_converter.py`**: (Runs on Dev PC)
        * Reads a `.json` sequence file (or all `.json` files in its directory).
        * Generates corresponding C++ function code for the *visual* sequence.
        * Automatically appends code to `sequences.cpp` and declarations to `sequences.hpp`.
        * Automatically updates `main.cpp` to add the sequence to the `setup()` list and the `loop()` command dispatch.
        * Automatically updates `controller.py` to list the new sequence name.
        * Triggers PlatformIO to compile and upload the updated firmware to the ESP32.
    * **`controller.py`**: (Runs on Dev PC or Pi)
        * Connects to the ESP32 via USB Serial (port configured via `config.ini`).
        * Provides a command-line interface to send `RUN:<sequence_name>` and `STOP` commands.
        * When a `RUN` command is issued:
            * Sends the command to the ESP32 to start the light sequence.
            * Automatically searches for a matching audio file (`<sequence_name>.wav`, `.flac`, or `.mp3`) in its directory.
            * If found, uses the `AudioPlayer` class (leveraging PyAudio/ffplay) to play the audio file, synchronized with the start command sent to the ESP32.

---

## Workflow

1. **Design Sequence:** Use the `sequence_editor.py` GUI on your development PC to create a sequence. Configure visual parameters (frequencies, brightness ramps, waveforms) and **audio parameters** using the built-in voice editor. Save the sequence as a `.json` file (e.g., `my_sequence.json`). Optionally generate the corresponding `.wav` file using the "Generate WAV" button in the GUI (which utilizes `sound_creator.py`).
2. **Prepare Files:** Place the saved `.json` file and the generated `.wav` (or other matching audio file - `.flac`, `.mp3`) into the same directory as the `json_to_cpp_converter.py` and `controller.py` scripts.
3. **Convert & Upload (on Dev PC):** Run the `json_to_cpp_converter.py` script. It will automatically:
    * Find `my_sequence.json` (and any other `.json` files).
    * Generate C++ code (`void my_sequence() { ... }`) for the **visual** part.
    * Append the code/declaration to `sequences.cpp`/`.hpp`.
    * Update `main.cpp` and `controller.py`.
    * Compile and upload the firmware to the connected ESP32 via PlatformIO.
    *(Ensure `setup.py` has been run previously to configure paths).*
4. **Run Sequence (on Dev PC or Pi):**
    * Ensure the ESP32 is connected via USB.
    * Run the `controller.py` script.
    * At the `>` prompt, type `RUN:my_sequence` and press Enter.
    * The script sends the command to the ESP32 (starting the lights) and simultaneously starts playing `my_sequence.wav` (or `.flac`/`.mp3` if found).
    * Type `STOP` to tell the ESP32 to stop the light sequence (audio playback on the host will need separate handling or stop when the script exits/is interrupted). Type `EXIT` to close the controller script.
