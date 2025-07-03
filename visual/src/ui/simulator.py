import json
import math
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QRectF # Removed QUrl
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont

# REMOVED: QtMultimedia imports and HAS_MULTIMEDIA flag

class LEDDisplay(QWidget):
    """
    A custom widget that displays 6 LEDs arranged in a hexagon.
    Each LED is drawn as a circle whose brightness (0.0 to 1.0) is updated.
    Even-indexed LEDs are “cool white” and odd-indexed LEDs are “warm white.”
    Below each LED a text label shows its instantaneous frequency and intensity.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.led_brightness = [0.0] * 6  # brightness values for each LED
        self.led_params = [""] * 6       # parameter strings for each LED
        # Define base colors.
        self.cool_white = QColor(240, 240, 240)  # light cool white
        self.warm_white = QColor(255, 223, 196)   # soft warm white
        self.setMinimumSize(300, 300)

    def update_leds(self, brightness_list):
        """Update the brightness values and refresh the display."""
        # Ensure list has correct length, clamp values
        valid_brightness = [max(0.0, min(1.0, b)) for b in brightness_list[:6]]
        # Pad if list is too short
        while len(valid_brightness) < 6:
            valid_brightness.append(0.0)
        self.led_brightness = valid_brightness
        self.update()


    def update_params(self, params_list):
        """Update the parameter strings for each LED."""
         # Ensure list has correct length
        valid_params = list(params_list[:6])
        # Pad if list is too short
        while len(valid_params) < 6:
            valid_params.append("")
        self.led_params = valid_params
        self.update()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", 10)
        painter.setFont(font)

        # Center and radius for the hexagon.
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        # Make hex_radius slightly smaller relative to window size for text room
        hex_radius = min(width, height) / 2 * 0.6
        led_radius = max(10, min(width, height) / 15) # Scale LED radius

        # Compute positions: 6 points evenly distributed, starting at 90° (12:00)
        positions = []
        for i in range(6):
            angle_deg = 90 - i * 60
            angle_rad = math.radians(angle_deg)
            x = center_x + hex_radius * math.cos(angle_rad)
            y = center_y - hex_radius * math.sin(angle_rad) # Y decreases upwards
            positions.append((x, y))

        # Draw each LED.
        for i, (x, y) in enumerate(positions):
            # Choose base color based on alternating pattern.
            base_color = self.cool_white if i % 2 == 0 else self.warm_white
            brightness = self.led_brightness[i]
            # Interpolate color: multiply each RGB component by brightness.
            try:
                 color = QColor(
                    int(base_color.red() * brightness),
                    int(base_color.green() * brightness),
                    int(base_color.blue() * brightness)
                )
            except Exception as e:
                 print(f"Error calculating color for LED {i}: {e}")
                 color = QColor(0, 0, 0) # Default to black on error

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen) # No border for the LED circle itself
            painter.drawEllipse(QRectF(x - led_radius, y - led_radius, led_radius * 2, led_radius * 2))

            # Draw outline for visibility when off
            painter.setPen(QColor(150, 150, 150)) # Light gray outline
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(x - led_radius, y - led_radius, led_radius * 2, led_radius * 2))


            # Draw the parameter text below the LED.
            text = self.led_params[i]
            # Use QFontMetrics for accurate centering
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(text)
            text_height = fm.height()
            painter.setPen(Qt.black) # Black text
             # Position text centered below the LED circle
            text_x = x - text_width / 2
            text_y = y + led_radius + text_height # Position below circle + gap + text height
            painter.drawText(int(text_x), int(text_y), text)



class SimulatorWindow(QMainWindow):
    """
    A simulator window that reads a sequence JSON.
    It visually replicates the LED sequence with blinking and parameter indicators,
    and provides playback controls. Audio playback is removed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LED Sequence Simulator")
        self.resize(600, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # LED display area.
        self.led_display = LEDDisplay()
        main_layout.addWidget(self.led_display)

        # Playback controls.
        controls_layout = QHBoxLayout()
        self.btn_load_sequence = QPushButton("Load Sequence JSON")
        # REMOVED: self.btn_load_audio
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        controls_layout.addWidget(self.btn_load_sequence)
        # REMOVED: controls_layout.addWidget(self.btn_load_audio)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_stop)
        main_layout.addLayout(controls_layout)

        # Progress slider and time label.
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000) # Default range, updated on load
        main_layout.addWidget(self.slider)
        self.lbl_time = QLabel("0.0 / 0.0 sec")
        main_layout.addWidget(self.lbl_time, alignment=Qt.AlignCenter) # Center label

        # Connect buttons.
        self.btn_load_sequence.clicked.connect(self.load_sequence)
        # REMOVED: self.btn_load_audio.clicked.connect(self.load_audio)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.slider_value_changed) # Update time label smoothly

        # Timer for updating simulation.
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # roughly 30 fps (33ms interval)
        self.timer.timeout.connect(self.update_simulation)

        # Simulation state.
        self.sequence_steps = None # Store only the steps list
        self.total_duration = 0.0
        self.current_time = 0.0
        self.is_playing = False

        # REMOVED: self.audio_player instance

        # Initial state of controls
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider.setEnabled(False)


    def load_sequence(self):
        """Load the sequence JSON file and compute total duration."""
        if self.is_playing:
             self.pause() # Pause if playing before loading

        fname, _ = QFileDialog.getOpenFileName(self, "Load Sequence JSON", "", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r') as f:
                    data = json.load(f)
                # Expect a JSON structure with a "steps" key.
                loaded_steps = data.get("steps", [])
                if not isinstance(loaded_steps, list):
                     raise ValueError("JSON 'steps' key does not contain a list.")

                self.sequence_steps = loaded_steps # Store the raw steps list
                if not self.sequence_steps:
                    QMessageBox.warning(self, "Empty Sequence", "The loaded sequence JSON contains no steps.")
                    # Reset state if sequence is empty
                    self.total_duration = 0.0
                    self.current_time = 0.0
                    self.btn_play.setEnabled(False)
                    self.slider.setEnabled(False)
                else:
                    # Sum durations. Handle potential missing/invalid duration.
                    self.total_duration = sum(step.get("duration", 0) for step in self.sequence_steps if isinstance(step, dict))
                    if self.total_duration <= 0:
                        QMessageBox.warning(self, "Invalid Duration", "Total sequence duration is zero or negative. Cannot play.")
                        self.total_duration = 0.0
                        self.btn_play.setEnabled(False)
                        self.slider.setEnabled(False)
                    else:
                        self.btn_play.setEnabled(True)
                        self.slider.setEnabled(True)

                # Reset time and slider
                self.current_time = 0.0
                slider_max = int(self.total_duration * 1000) if self.total_duration > 0 else 1000
                self.slider.setRange(0, slider_max)
                self.slider.setValue(0)
                self.update_time_label() # Use helper
                self.update_led_display() # Show first frame
                self.btn_pause.setEnabled(False)
                self.btn_stop.setEnabled(False)

            except json.JSONDecodeError as e:
                 QMessageBox.critical(self, "JSON Error", f"Failed to parse sequence JSON:\n{e}")
                 self.reset_simulator_state()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load or process sequence JSON:\n{e}")
                self.reset_simulator_state()


    # REMOVED: load_audio() method

    def reset_simulator_state(self):
        """ Resets the simulator to a default unloaded state """
        self.stop() # Ensure timer is stopped etc.
        self.sequence_steps = None
        self.total_duration = 0.0
        self.current_time = 0.0
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.update_time_label()
        self.led_display.update_leds([0.0] * 6)
        self.led_display.update_params([""] * 6)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider.setEnabled(False)

    def play(self):
        """Start the simulation."""
        if self.sequence_steps is None or self.total_duration <= 0:
            # Should not be possible if button is correctly enabled/disabled, but check anyway
            QMessageBox.warning(self, "Error", "Please load a valid sequence JSON first.")
            return

        if not self.is_playing:
             # If starting from beginning or resuming after stop
            if self.current_time >= self.total_duration:
                self.current_time = 0.0 # Restart if at end

            self.timer.start()
            self.is_playing = True
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_load_sequence.setEnabled(False) # Disable loading while playing


    def pause(self):
        """Pause the simulation."""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(True) # Stop is always available when paused
            self.btn_load_sequence.setEnabled(True) # Allow loading when paused

    def stop(self):
        """Stop playback and reset time."""
        self.timer.stop()
        self.is_playing = False
        self.current_time = 0.0
        self.slider.setValue(0)
        self.update_time_label()
        self.update_led_display() # Update LEDs to time 0 state
        self.btn_play.setEnabled(self.sequence_steps is not None and self.total_duration > 0) # Re-enable play if sequence loaded
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False) # Stop is not needed when already stopped
        self.btn_load_sequence.setEnabled(True)

        # REMOVED: self.audio_player.stop()

    def slider_moved(self, value):
        """Handle manual slider movement while dragging."""
        if not self.timer.isActive(): # Only update time if paused or stopped
            self.current_time = value / 1000.0
            # Clamp time to bounds
            self.current_time = max(0.0, min(self.current_time, self.total_duration))
            self.update_time_label()
            self.update_led_display()
            # REMOVED: self.audio_player.setPosition(value)

    def slider_value_changed(self, value):
        """Handle slider value change (e.g., from timer or click)."""
        # This updates the label smoothly even when playing
        current_slider_time = value / 1000.0
        self.lbl_time.setText(f"{current_slider_time:.1f} / {self.total_duration:.1f} sec")


    def update_time_label(self):
        """ Helper to update the time label based on current_time """
        self.lbl_time.setText(f"{self.current_time:.1f} / {self.total_duration:.1f} sec")


    def update_simulation(self):
        """Timer callback to update the current time and LED display."""
        delta_time = self.timer.interval() / 1000.0
        self.current_time += delta_time

        if self.current_time >= self.total_duration:
            self.current_time = self.total_duration # Clamp to end
            self.stop() # Stop when reaching the end
            # Set slider precisely to max value at the end
            self.slider.setValue(self.slider.maximum())
            self.update_time_label()
            self.update_led_display() # Ensure last frame is displayed correctly
            return

        # Update slider position smoothly
        self.slider.setValue(int(self.current_time * 1000))
        # Time label is updated by slider_value_changed signal

        self.update_led_display()


    def update_led_display(self):
        """
        Update LED brightness and parameter indicators for the current simulation time.
        Blinking is driven by the oscillator frequency while intensity is computed from strobe_sets.
        Parameter indicators show the instantaneous frequency and the base intensity (in %).
        """
        if self.sequence_steps is None:
            self.led_display.update_leds([0.0] * 6)
            self.led_display.update_params([""] * 6)
            return

        elapsed = self.current_time
        step_start_time = 0.0
        current_step_data = None
        time_within_step = 0.0

        for step in self.sequence_steps:
             # Basic validation
            if not isinstance(step, dict): continue
            duration = step.get("duration", 0)
            if duration <= 0: continue # Skip steps with no duration

            step_end_time = step_start_time + duration
            # Use a small tolerance for floating point comparisons
            epsilon = 1e-6
            if elapsed >= step_start_time - epsilon and elapsed < step_end_time - epsilon:
                current_step_data = step
                time_within_step = elapsed - step_start_time
                break
             # Handle reaching the exact end of the last step
            elif abs(elapsed - step_end_time) < epsilon and step == self.sequence_steps[-1]:
                 current_step_data = step
                 # Represent being exactly at the end by setting time_within_step to duration
                 time_within_step = duration
                 break

            step_start_time += duration


        brightness = [0.0] * 6
        params = [""] * 6

        if current_step_data is None:
             # If somehow time is outside all steps (shouldn't happen if logic is right)
             # Or if sequence just finished, keep LEDs off.
            self.led_display.update_leds(brightness)
            self.led_display.update_params(params)
            return

        # Get step properties safely
        step_duration = current_step_data.get("duration", 1.0) # Avoid division by zero
        if step_duration <= 0: step_duration = 1.0 # Safety fallback

        # Interpolation factor within the step (0.0 to 1.0)
        interp = time_within_step / step_duration
        interp = max(0.0, min(1.0, interp)) # Clamp interpolation factor


        oscillators_data = current_step_data.get("oscillators", [])
        strobe_sets_data = current_step_data.get("strobe_sets", [])

        num_oscillators = len(oscillators_data)
        num_strobe_sets = len(strobe_sets_data)

        # --- Determine Mode based on counts (more robustly) ---
        mode = "Unknown"
        if num_oscillators == 1 and num_strobe_sets == 1:
            mode = "Combined"
        elif num_oscillators == 2 and num_strobe_sets == 2:
            mode = "Split"
        elif num_oscillators == 6 and num_strobe_sets == 6:
            mode = "Independent"
        else:
            # Handle inconsistent data - maybe default to Combined or log error
             print(f"Warning: Inconsistent oscillator ({num_oscillators}) / strobe set ({num_strobe_sets}) counts. Simulating based on available data.")
             # We'll proceed, but results might be weird.


        # --- Calculate Instantaneous Values ---
        inst_osc_values = [] # Store (freq, duty, waveform_type, phase_offset, brightness_mod) tuples

        for i in range(num_oscillators):
            osc_data = oscillators_data[i]
            if not isinstance(osc_data, dict): continue # Skip invalid osc data

            f0 = osc_data.get("start_freq", 0.0)
            f1 = osc_data.get("end_freq", f0) # Default end freq to start freq
            d0 = osc_data.get("start_duty", 50.0)
            d1 = osc_data.get("end_duty", d0)
            waveform_type = osc_data.get("waveform", 0) # 0:Off, 1:Square, 2:Sine

            inst_freq = f0 + (f1 - f0) * interp
            inst_duty = d0 + (d1 - d0) * interp # Duty cycle 0-100

            # Calculate integrated phase: phase = integral(2*pi*f(t) dt) from 0 to time_within_step
            # For linear frequency change f(t) = f0 + (f1-f0)/T * t
            # Integral(f(t) dt) = f0*t + 0.5*(f1-f0)/T * t^2
            phase = 2 * math.pi * (f0 * time_within_step + 0.5 * (f1 - f0) / step_duration * time_within_step**2)

             # TODO: Add RFM simulation if needed (would require state tracking)
             # TODO: Add Pattern simulation if needed (complex, requires specific formulas)
            phase_offset = 0.0 # Placeholder for phase patterns
            brightness_mod = 1.0 # Placeholder for brightness patterns


            inst_osc_values.append((inst_freq, inst_duty, waveform_type, phase + phase_offset, brightness_mod))


        # --- Calculate LED Brightness ---
        final_brightness = [0.0] * 6
        final_params = [""] * 6

        for sset_idx, sset_data in enumerate(strobe_sets_data):
            if not isinstance(sset_data, dict): continue

            led_indices = sset_data.get("channels", [])
            i0 = sset_data.get("start_intensity", 0.0)
            i1 = sset_data.get("end_intensity", i0)
            weights = sset_data.get("oscillator_weights", [])

            inst_intensity_base = i0 + (i1 - i0) * interp # Intensity 0-100

            # Weighted sum of oscillator outputs
            oscillator_contribution = 0.0
            active_osc_found = False

            for osc_idx, weight in enumerate(weights):
                 if weight > 0 and osc_idx < len(inst_osc_values):
                    freq, duty, wave_type, phase, bright_mod = inst_osc_values[osc_idx]
                    osc_output = 0.0 # Value from 0.0 to 1.0 based on waveform

                    if wave_type == 1: # Square
                        # Phase wraps at 2*pi. Check if within the 'on' portion.
                        duty_fraction = max(0.01, min(0.99, duty / 100.0)) # Clamp duty 1-99%
                        if (phase % (2 * math.pi)) < (2 * math.pi * duty_fraction):
                            osc_output = 1.0
                    elif wave_type == 2: # Sine
                        # Map [-1, 1] sine output to [0, 1] brightness
                        osc_output = (math.sin(phase) + 1.0) / 2.0
                    # wave_type 0 (Off) results in osc_output = 0.0

                    oscillator_contribution += weight * osc_output * bright_mod # Apply brightness mod
                    active_osc_found = True


            # If no active oscillators contribute, contribution remains 0
            # Normalize intensity base to 0.0-1.0
            intensity_factor = max(0.0, min(1.0, inst_intensity_base / 100.0))

            # Calculate final brightness for LEDs in this set
            calculated_brightness = intensity_factor * oscillator_contribution

            # Assign to relevant LEDs
            for led_idx in led_indices:
                 if 0 <= led_idx < 6:
                    final_brightness[led_idx] = calculated_brightness

                    # Determine which oscillator primarily drives this LED for parameter display
                    # Find oscillator with max weight for this strobe set
                    primary_osc_idx = -1
                    max_weight = -1
                    if weights:
                         max_weight = max(weights)
                         if max_weight > 0:
                              primary_osc_idx = weights.index(max_weight)

                    # Get params from the primary oscillator if found
                    if primary_osc_idx != -1 and primary_osc_idx < len(inst_osc_values):
                         freq, duty, _, _, _ = inst_osc_values[primary_osc_idx]
                         final_params[led_idx] = f"{freq:.1f}Hz, {inst_intensity_base:.0f}%"
                    else:
                         # Fallback if no weights or primary osc cannot be determined
                         final_params[led_idx] = f"?, {inst_intensity_base:.0f}%"


        self.led_display.update_leds(final_brightness)
        self.led_display.update_params(final_params)


    def closeEvent(self, event):
        """Ensure timer is stopped when the window closes."""
        self.timer.stop()
        # REMOVED: self.audio_player.stop()
        event.accept()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = SimulatorWindow()
    window.show()
    sys.exit(app.exec_())
