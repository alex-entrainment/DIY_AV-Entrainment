# DIY Audio-Visual Brainwave Entrainment

This project aims to create an open-source audio-visual brainwave entrainment system using a small single-board computer (SBC) such as a Raspberry Pi.  

It drives 6 high-power LEDs via a PCA9685 PWM board and MOSFETs, and optionally generates and plays synchronized audio entrainment tracks (e.g. binaural, isochronic).  

Users can design custom multi-step “sequences” (lighting + audio parameters) with a PyQt-based GUI editor, then run them on hardware.

Total Device Cost: ~$150

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Components](#hardware-components)
4. [Tools Required](#tools-required)
5. [System Architecture](#system-architecture)
7. [Installation](#installation)
8. [GUI Overview](#gui-overview)
9. [Resources](#resources)
---

## Overview
Brainwave entrainment (BWE) involves the use of pulsing lights or audio tones at specific frequencies in an attempt to influence brain states. This project allows you to experiment with various stimulation patterns:
- **Visual**: 6 strobing LEDs with adjustable waveforms, duty cycles, frequency ramps, and more.
- **Audio**: Up to 3 carriers that can be individually enabled, each with optional frequency ramps, random frequency modulation (RFM), binaural or isochronic modes, and pink noise.

The result is a flexible platform for exploring entrainment principles. The system is intended for DIY research and is **not** a medical device. Build and use at your own discretion.

![device front](https://github.com/user-attachments/assets/9552d6c8-8354-40ea-99f3-c1e1ff4e83d6)
![device back](https://github.com/user-attachments/assets/c19d92c4-99ef-4471-9ec9-e069b728cf6b)

![20250308_190018](https://github.com/user-attachments/assets/f11bc455-edf9-4216-9470-9396f6ea62ef)



---

## Features
- **PyQt5-based GUI** (in `sequence_editor.py`) for creating multi-step LED + audio sequences.
- **Advanced Oscillator Patterns (work in progress)**: Sine, square, and configurable “phase/brightness/timing” secondary patterns (e.g., “Sacred Geometry,” “Fractal Arc,” etc.).
- **Multiple Oscillator Control Modes**: Combined (single oscillator), Split (cool/warm oscillators), and Independent (6 independent oscillators) modes available for maximum sequence customizability.
- **Multi-Step Sequencing**: Generates multi-stage LED pulse sequences 
- **Random Frequency Generation**: Configurable, slight changes to the binaural or visual pulse frequency to reduce habituation to a steady-state stimulus. 
- **Binaural Generation**: Can generate a matching, stepped binaural .wav file, synced to the LED "track" with up to 3 simultaneous binaurals. Supports isochronic tones + pink noise. Plays audio automatically when run on-device, if present. 
- **Linear Ramps**: Frequencies or duty cycles can transition over the duration of each step.
- **Strobe Intensity**: Per-step strobe intensity or crossfade.
- **JSON File Storage**: Save or load complete sequences, including audio settings, to a `.json` file.
- **Stepwise Audio Generation**: Automatic `.wav` generation using the duration and frequency ramps of each step, as configured.

---

## Hardware Components
Below is the minimal parts list used for the LED system:

1. **SBC computer** (Raspberry Pi or comparable)
 - [Raspberry Pi 3B](https://www.digikey.com/en/products/detail/raspberry-pi/SC0073/8571724?gclsrc=aw.ds&&utm_adgroup=&utm_source=google&utm_medium=cpc&utm_campaign=PMax%20Shopping_Product_Medium%20ROAS%20Categories&utm_term=&utm_content=&utm_id=go_cmp-20223376311_adg-_ad-__dev-c_ext-_prd-8571724_sig-Cj0KCQiA2oW-BhC2ARIsADSIAWpJQ6Hkttar9WaTSVHl90eo3DTDA_QYNAuV6JZE0Mu0xg5CFHhTvtcaAv_WEALw_wcB&gad_source=1&gclid=Cj0KCQiA2oW-BhC2ARIsADSIAWpJQ6Hkttar9WaTSVHl90eo3DTDA_QYNAuV6JZE0Mu0xg5CFHhTvtcaAv_WEALw_wcB&gclsrc=aw.ds)
   
2. **PCA9685** 16-channel PWM driver
  - [PCA9685](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwj8nsrUs-eLAxXGWEcBHZsXJv0YABASGgJxdQ&co=1&gclid=Cj0KCQiA2oW-BhC2ARIsADSIAWqau1z6inqUk-G94FxrL8oTb7Xaf7VOVPfy7Y1CcpPGAOvCtB1W3F8aAlbkEALw_wcB&ohost=www.google.com&cid=CAESVuD2YdQKXnas6BOsyXCBJwyZxfZIIJkp21_CLLfYlzfvBrurVr-enGlMwICQPxCyaJH64vcYN6Xl2ZUs1XS7ggdYaPj_Qac4AeXundOGM28ghOi9fyXC&sig=AOD64_2lIttoYyqeN1kmbOn14SrPFssUmQ&ctype=5&q=&ved=2ahUKEwiX3cPUs-eLAxWrD1kFHTLPFVIQ9aACKAB6BAgLEBA&adurl=)
    
3. **3× High-power (~0.3 A × 3.3 V) cool white LEDs**
   - [Uxcell Cool White LEDs](https://www.amazon.com/dp/B07DHB61BH?ref=ppx_yo2ov_dt_b_fed_asin_title)

4. **3× High-power warm white LEDs**
   - [Uxcell Warm White LEDs](https://www.amazon.com/dp/B07DHB13J4?ref_=ppx_hzod_title_dt_b_fed_asin_title_0_3)
     
5. **6× Switching MOSFETs** (e.g., TIP120, IRLZ44N w/ 5v logic-level shift; or similar)  
   *(If they are not fully logic-level at 3.3 V, a 5 V level shifter or gate driver may be required)*
   - [TIP 120 transistors](https://www.adafruit.com/product/976)
     
6. **1× External 5 V power supply** (capable of driving the 6 LEDs + the Raspberry Pi if desired)
   - [5V 10A external power supply (barrel connector)](https://www.adafruit.com/product/658)
     
7. **6× 1-2 W, 5 Ω resistors** (one per LED for current limiting)
   - [30x 5.1ohm 2W resistors](https://www.amazon.com/Resistor-Tolerance-Resistors-Limiting-Certificated/dp/B08QRCRKHR/ref=sr_1_3?crid=2DBPEC5KWMJRB&dib=eyJ2IjoiMSJ9.CXcWo8xeOs8uHGebOYTuZn8BIX37Om6YoXE5yIfUyxO5E9T4DC4SMuOyTKehtTSCsE577x43yWjVL28fDDgO0nqexY786KxTFwEA_NHCsBUt0LsP_D-xYa9Mt3YRFVSHCIm7ed4lwgx5J2fkEncPBigRNCpgX77eFLdPsh5YE04i6nzXqbBcFRkUS-ChnKpF9L5lAX2skqe5RsxzaJ47W5BSF-Zjt9uv7hZittCKuZ-98nQNRlXmCPvYnG9FTnHnzMSb2PMtjBjgEXUcphOf0QLGf397FXS6dnyPXHthXqwEsSjtnpw7eg53hDDbqbFnXHr3OtP7jkiGco60zwsOkkIIcpGY4kajyed4_kYQ9Ko.hBNuoHMv5EH1yYzY1y5gi3bOq40UKEIFvAKwJ-DJh8g&dib_tag=se&keywords=5.1ohm+1.5W&qid=1740784354&s=industrial&sprefix=5.1ohm+1.5w%2Cindustrial%2C71&sr=1-3)
     
8. **6x 10KΩ resistors** (pull-downs for MOSFET gates)
   - [10kohm resistors](https://www.digikey.com/en/products/detail/koa-speer-electronics-inc/CFP1-4CT52R103J/13538361?gclsrc=aw.ds&&utm_adgroup=General&utm_source=google&utm_medium=cpc&utm_campaign=PMax%20Shopping_Product_Zombie%20SKUs&utm_term=&utm_content=General&utm_id=go_cmp-17815035045_adg-_ad-__dev-c_ext-_prd-13538361_sig-Cj0KCQiA2oW-BhC2ARIsADSIAWrdYJ9RNXP3d_Qc8jxdgaeR2kErVAGhhzi6obiyrM_jp4Z8ABrjA9caAg3UEALw_wcB&gad_source=1&gclid=Cj0KCQiA2oW-BhC2ARIsADSIAWrdYJ9RNXP3d_Qc8jxdgaeR2kErVAGhhzi6obiyrM_jp4Z8ABrjA9caAg3UEALw_wcB&gclsrc=aw.ds)
     
10. **6× 220 Ω resistors** (protects PWM pins)
  - [220Ohm Resistors](https://www.digikey.com/en/products/detail/stackpole-electronics-inc/CF14JT220K/1741348)
   
12. **Wiring** ( 22AWG jumper wires, hookup wire, etc.)
  - [Jumper Wires](https://www.amazon.com/dp/B01EV70C78?ref=ppx_yo2ov_dt_b_fed_asin_title)

13. **Device Stand**
  - [Sound Percussion Labs KBS200 Endeavor Series Double-Braced Cymbal Boom Stand](https://www.guitarcenter.com/Sound-Percussion-Labs/KBS200-Double-Braced-Cymbal-Boom-Stand-1500000405266.gc?template=0y7n73MAL4Km&storeCode=&source=4WWRWXGL&utm_medium=paid-search&utm_channel=paid-search&utm_source=google&utm_platform=google&utm_campaign=GC_G_NTM_LIA-PMX_Accessories_N&utm_ct=lia&utm_tactic=prospecting&utm_segment=accessory&utm_term=&utm_content=119280402&gad_source=1&gclid=Cj0KCQiAoJC-BhCSARIsAPhdfSh7YFz3OY7G7pWrZ0CywWRpzmroID6TKPOMfr9qcrGHxDkLxJ6iitsaAr9UEALw_wcB)

### Typical Connections
- The PCA9685 connects via I²C (SDA, SCL) to the SBC (Raspberry Pi pins).
- PCA9685 outputs go to the MOSFET gates (with pull-down resistors to ensure they remain off when idle).
- MOSFET drains connect to each LED + current-limiting resistor in series, then to the 5 V supply rail.
- MOSFET sources go to ground.
- Make sure to share a common ground between the SBC, PCA9685, and the external 5 V supply.

---

## Tools Required
1. **Soldering iron + flux**
2. **Wire strippers**
3. **Multimeter** (optional but recommended for verifying current/voltage)
4. **Breadboard** (optional for prototyping)
5. **3D printer** (optional if you want a custom LED holder or enclosure)

##  Circuit Diagram
![image](https://github.com/user-attachments/assets/84c5e81f-429e-4560-a3dc-5627ef1da878)


---

## System Architecture
1. **`sequence_editor.py` (PyQt5 GUI)**:
   - Allows you to add/duplicate/remove steps and configure oscillator waveforms, frequencies, random modulation (RFM), and more for each LED group or individual LED.
   - Manages global audio settings such as binaural mode, isochronic pulses, multiple carriers, pink noise, etc.
   - Saves the sequence + settings to JSON.
   - Optionally generates a `.wav` audio file matching the final sequence’s durations and frequencies.

2. **`sequence_model.py`**:
   - Contains the data classes (`Step`, `Oscillator`, `StrobeSet`, `AudioSettings`, etc.) that define how the LED and audio steps are stored, serialized, and reconstructed from JSON.

3. **`audio_generator.py`**:
   - Functions that generate mono or stereo waveforms based on your audio carrier setups.
   - Handles random frequency modulation, pink noise, binaural or isochronic calculations, etc.
   - Capable of stitching multiple step waveforms into a single `.wav` file for the entire sequence.

4. **`run_on_device_6.py`**:
   - A command-line runner that loads a JSON sequence (created by the editor) and physically drives the PCA9685 + LEDs, optionally playing the generated `.wav` file in sync.
   - Applies real-time oscillator logic: waveforms, frequency ramps, RFM, brightness patterns, etc.
   - To start running a sequence from a file: `python run_on_device_6.py --file "filename".json`

---

## Installation
1. **Clone or download** this repository onto your development machine and/or Raspberry Pi.
2. Install required Python packages: 
   numpy – for audio waveform generation
   simpleaudio – for real-time audio playback
   adafruit-circuitpython-pca9685 – for driving the PCA9685 over I²C
   PyQt5 – for the GUI editor
   You may also need PyQt5-sip and other standard libraries for your Python environment.
   Enable I²C on your Raspberry Pi (if using Pi OS), then wire the PCA9685 accordingly.
   

## Usage 
### Commands (run_on_device_6.py): 
      1. --file <filename> // Selects the JSON sequence file to run
      2. --start-from <time> // Starts JSON sequence at <time> 

## GUI Overview

Below is a screenshot of the Sequence Editor GUI in "Split" mode (PyQt5):
![image](https://github.com/user-attachments/assets/4cd68e9d-2334-479b-8db8-0e1d32ed64a7)




Left Panel: List of steps in the sequence. You can Add, Duplicate, Remove, or Reorder steps.

Step Info: Set the duration of the step and a short description.

Oscillator Mode:
- Combined: 1 oscillator controlling all 6 LEDs.
- Split: 2 oscillators (e.g., one for even-indexed LEDs, one for odd-indexed).
- Independent: Each LED has its own oscillator.

Oscillator Settings: Choose waveform (Sine, Square, Off), frequency start/end, duty cycle start/end, random frequency modulation (RFM) parameters, and advanced pattern modulation.

Strobe Intensities: Overall intensity ramp for each group of LEDs.

Audio Tabs (right side): Up to 3 carriers, each with optional frequency ramps, RFM, volume control. The Global Settings tab includes enabling audio, binaural or isochronic mode, pink noise, etc.
---
## Resources

- [Creating your own Light Sequences](https://support.pandorastar.co.uk/wp-content/uploads/sites/6/2021/01/Creating-your-own-sequences-on-PandoraStar-1.pdf)
- [List of proprietary programs from PandoraStar, useful for examples](https://support.pandorastar.co.uk/wp-content/uploads/sites/6/2021/01/PS-Program-List-2020.pdf)
- [Gnaural binaural / isochronic tone creation software](https://gnaural.sourceforge.net/)

- [Research on photic entrainment / driving](https://indyneurofeedback.com/wp-content/uploads/2018/02/Article-1-AVE-History-and-Physiological-Mechanisms.pdf#:~:text=%28Joyce%20%26%20Siever%2C%202000%3B%20Ruuskanen,1973%3B%20Regan%2C%201966%3B%20Siever%2C%202002)
- 


