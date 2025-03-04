
import json
import math
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont

# Try to import multimedia support.
try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtCore import QUrl
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False

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
        self.warm_white = QColor(255, 223, 196)    # soft warm white
        self.setMinimumSize(300, 300)

    def update_leds(self, brightness_list):
        """Update the brightness values and refresh the display."""
        self.led_brightness = brightness_list
        self.update()

    def update_params(self, params_list):
        """Update the parameter strings for each LED."""
        self.led_params = params_list
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
        hex_radius = min(width, height) / 2 * 0.8  # radius for LED placement
        led_radius = 20

        # Compute positions: 6 points evenly distributed, starting at 90° (12:00)
        positions = []
        for i in range(6):
            angle_deg = 90 - i * 60
            angle_rad = math.radians(angle_deg)
            x = center_x + hex_radius * math.cos(angle_rad)
            y = center_y - hex_radius * math.sin(angle_rad)
            positions.append((x, y))

        # Draw each LED.
        for i, (x, y) in enumerate(positions):
            # Choose base color based on alternating pattern.
            base_color = self.cool_white if i % 2 == 0 else self.warm_white
            brightness = self.led_brightness[i]
            # Interpolate color: multiply each RGB component by brightness.
            color = QColor(
                int(base_color.red() * brightness),
                int(base_color.green() * brightness),
                int(base_color.blue() * brightness)
            )
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(x - led_radius, y - led_radius, led_radius * 2, led_radius * 2))
            # Draw the parameter text below the LED.
            text = self.led_params[i]
            text_width = painter.fontMetrics().width(text)
            painter.setPen(Qt.black)
            painter.drawText(int(x - text_width / 2), int(y + led_radius + 15), text)

class SimulatorWindow(QMainWindow):
    """
    A simulator window that reads a sequence JSON and (optionally) a matching audio WAV file.
    It visually replicates the LED sequence with blinking and parameter indicators,
    and provides playback controls.
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
        self.btn_load_audio = QPushButton("Load Audio WAV")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        controls_layout.addWidget(self.btn_load_sequence)
        controls_layout.addWidget(self.btn_load_audio)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_stop)
        main_layout.addLayout(controls_layout)

        # Progress slider and time label.
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        main_layout.addWidget(self.slider)
        self.lbl_time = QLabel("0.0 / 0.0 sec")
        main_layout.addWidget(self.lbl_time)

        # Connect buttons.
        self.btn_load_sequence.clicked.connect(self.load_sequence)
        self.btn_load_audio.clicked.connect(self.load_audio)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.slider.sliderMoved.connect(self.slider_moved)

        # Timer for updating simulation.
        self.timer = QTimer(self)
        self.timer.setInterval(30)  # roughly 33 fps
        self.timer.timeout.connect(self.update_simulation)

        # Simulation state.
        self.sequence = None
        self.total_duration = 0.0
        self.current_time = 0.0

        # Audio player (if multimedia support is available).
        if HAS_MULTIMEDIA:
            self.audio_player = QMediaPlayer()
        else:
            self.audio_player = None

    def load_sequence(self):
        """Load the sequence JSON file and compute total duration."""
        fname, _ = QFileDialog.getOpenFileName(self, "Load Sequence JSON", "", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r') as f:
                    data = json.load(f)
                # Expect a JSON structure with a "steps" key.
                self.sequence = data.get("steps", [])
                if not self.sequence:
                    QMessageBox.warning(self, "Error", "No steps found in the sequence JSON.")
                    return
                # Sum durations.
                self.total_duration = sum(step.get("duration", 0) for step in self.sequence)
                self.current_time = 0.0
                self.slider.setRange(0, int(self.total_duration * 1000))
                self.slider.setValue(0)
                self.lbl_time.setText(f"0.0 / {self.total_duration:.1f} sec")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load sequence JSON:\n{e}")

    def load_audio(self):
        """Load the matching WAV audio file for playback."""
        if not HAS_MULTIMEDIA:
            QMessageBox.warning(self, "Error", "Multimedia support is not available.")
            return
        fname, _ = QFileDialog.getOpenFileName(self, "Load Audio WAV", "", "WAV Files (*.wav)")
        if fname:
            url = QUrl.fromLocalFile(fname)
            media = QMediaContent(url)
            self.audio_player.setMedia(media)

    def play(self):
        """Start the simulation and audio playback."""
        if self.sequence is None:
            QMessageBox.warning(self, "Error", "Please load a sequence JSON first.")
            return
        self.timer.start()
        if self.audio_player:
            self.audio_player.play()

    def pause(self):
        """Pause the simulation and audio."""
        self.timer.stop()
        if self.audio_player:
            self.audio_player.pause()

    def stop(self):
        """Stop playback and reset time."""
        self.timer.stop()
        self.current_time = 0.0
        self.slider.setValue(0)
        self.lbl_time.setText(f"0.0 / {self.total_duration:.1f} sec")
        self.update_led_display()
        if self.audio_player:
            self.audio_player.stop()

    def slider_moved(self, value):
        """Handle manual slider movement."""
        self.current_time = value / 1000.0
        self.update_led_display()
        self.lbl_time.setText(f"{self.current_time:.1f} / {self.total_duration:.1f} sec")
        if self.audio_player:
            self.audio_player.setPosition(value)

    def update_simulation(self):
        """Timer callback to update the current time and LED display."""
        self.current_time += self.timer.interval() / 1000.0
        if self.current_time > self.total_duration:
            self.stop()
            return
        self.slider.setValue(int(self.current_time * 1000))
        self.lbl_time.setText(f"{self.current_time:.1f} / {self.total_duration:.1f} sec")
        self.update_led_display()

    def update_led_display(self):
        """
        Update LED brightness and parameter indicators for the current simulation time.
        Blinking is driven by the oscillator frequency while intensity is computed from strobe_sets.
        Parameter indicators show the instantaneous frequency and the base intensity (in %).
        """
        if self.sequence is None:
            return
        elapsed = self.current_time
        step_start = 0.0
        current_step = None
        for step in self.sequence:
            duration = step.get("duration", 0)
            if step_start + duration >= elapsed:
                current_step = step
                time_in_step = elapsed - step_start
                break
            step_start += duration

        brightness = [0.0] * 6
        params = [""] * 6
        if current_step is None:
            # End of sequence.
            self.led_display.update_leds(brightness)
            self.led_display.update_params(params)
            return

        T = current_step.get("duration", 1)
        interp = time_in_step / T

        # Determine oscillator for each LED.
        oscillators = current_step.get("oscillators", [])
        osc_list = []
        if oscillators:
            if len(oscillators) >= 6:
                osc_list = oscillators[:6]
            else:
                osc_list = [oscillators[0]] * 6
        else:
            osc_list = [{}] * 6  # fallback

        # Determine strobe intensity for each LED.
        strobe_sets = current_step.get("strobe_sets", [])
        intensities = [0.0] * 6
        if len(strobe_sets) == 1:
            intensity = strobe_sets[0].get("start_intensity", 0) + \
                        (strobe_sets[0].get("end_intensity", 0) - strobe_sets[0].get("start_intensity", 0)) * interp
            intensities = [intensity] * 6
        elif len(strobe_sets) == 2:
            for i in range(6):
                strobe = strobe_sets[0] if i % 2 == 0 else strobe_sets[1]
                intensities[i] = strobe.get("start_intensity", 0) + \
                                 (strobe.get("end_intensity", 0) - strobe.get("start_intensity", 0)) * interp
        elif len(strobe_sets) >= 6:
            for i in range(6):
                strobe = strobe_sets[i]
                intensities[i] = strobe.get("start_intensity", 0) + \
                                 (strobe.get("end_intensity", 0) - strobe.get("start_intensity", 0)) * interp
        else:
            intensities = [0.0] * 6

        # Compute blinking factor and parameters for each LED.
        for i in range(6):
            osc = osc_list[i]
            f0 = osc.get("start_freq", 0)
            f1 = osc.get("end_freq", 0)
            inst_freq = f0 + (f1 - f0) * interp
            # Integrated phase for blinking.
            phase = 2 * math.pi * (f0 * time_in_step + 0.5 * (f1 - f0) / T * time_in_step * time_in_step)
            blink = 1.0 if math.sin(phase) >= 0 else 0.0

            # Compute brightness for this LED.
            brightness[i] = (intensities[i] / 100.0) * blink

            # Prepare parameter string.
            params[i] = f"Freq: {inst_freq:.1f}Hz, Int: {intensities[i]:.0f}%"

        self.led_display.update_leds(brightness)
        self.led_display.update_params(params)

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = SimulatorWindow()
    window.show()
    sys.exit(app.exec_())

