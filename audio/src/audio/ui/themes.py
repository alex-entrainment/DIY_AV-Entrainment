from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from dataclasses import dataclass

@dataclass
class Theme:
    palette_func: callable
    stylesheet: str = ""

def dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    return palette

# Style sheet ensuring editable widgets use white text in the dark theme
GLOBAL_STYLE_SHEET_DARK = """

QTreeWidget {
    color: #ffffff;
}

"""
    
# Green cymatic theme derived from the example in README
GLOBAL_STYLE_SHEET_GREEN = """
/* Base Widget Styling */
QWidget {
    font-size: 10pt;
    background-color: #0a0a0a;
    color: #00ffaa;
    font-family: 'Consolas', 'Courier New', monospace;
}

/* Group Boxes */
QGroupBox {
    background-color: #1a1a1a;
    border: 1px solid rgba(0, 255, 136, 0.4);
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
}

QGroupBox::title {
    color: #00ffaa;
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
    background-color: #1a1a1a;
}

/* Push Buttons */
QPushButton {
    background-color: rgba(0, 255, 136, 0.25);
    border: 1px solid #00ff88;
    color: #00ffaa;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: rgba(0, 255, 136, 0.4);
    border: 1px solid #00ffcc;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
}

QPushButton:pressed {
    background-color: rgba(0, 255, 136, 0.6);
}

QPushButton:disabled {
    background-color: rgba(0, 136, 68, 0.2);
    border: 1px solid rgba(0, 255, 136, 0.2);
    color: rgba(0, 255, 136, 0.5);
}

/* Column Headers */
QHeaderView::section {
    background-color: #000000;
    color: #00ffaa;
}

QLineEdit, QComboBox, QSlider {
    background-color: #202020;
    border: 1px solid #555555;
    color: #ffffff;     /* use white text */
}
"""

# Light blue theme with a neutral light palette and blue highlights
GLOBAL_STYLE_SHEET_LIGHT_BLUE = """
QTreeWidget {
    color: #000000;
}
QLineEdit, QComboBox, QSlider {
    background-color: #ffffff;
    border: 1px solid #a0a0a0;
    color: #000000;
}
"""

# Material theme with teal and orange accents
GLOBAL_STYLE_SHEET_MATERIAL = """
QTreeWidget {
    color: #212121;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    padding-left: 8px;
    padding-right: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px 0 4px;
}
QPushButton {
    background-color: #009688;
    border: none;
    color: white;
    padding: 6px 16px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #26a69a;
}
QPushButton:pressed {
    background-color: #00796b;
}
QLineEdit, QComboBox, QSlider {
    background-color: #ffffff;
    border: 1px solid #bdbdbd;
    color: #212121;
    border-radius: 4px;
}
"""

def green_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(0x0a, 0x0a, 0x0a))
    palette.setColor(QPalette.WindowText, QColor(0x00, 0xff, 0xaa))
    palette.setColor(QPalette.Base, QColor(0x1a, 0x1a, 0x1a))
    palette.setColor(QPalette.AlternateBase, QColor(0x15, 0x20, 0x15))
    palette.setColor(QPalette.Text, QColor(0x00, 0xff, 0xaa))
    palette.setColor(QPalette.Button, QColor(0x00, 0x88, 0x44, 0x60))
    palette.setColor(QPalette.ButtonText, QColor(0x00, 0xff, 0xaa))
    palette.setColor(QPalette.Highlight, QColor(0x00, 0xff, 0x88, 0xaa))
    palette.setColor(QPalette.HighlightedText, QColor(0xff, 0xff, 0xff))
    palette.setColor(QPalette.Link, QColor(0x00, 0xff, 0xcc))
    return palette

def light_blue_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 248, 255))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(230, 240, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(225, 238, 255))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 122, 204))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette

def material_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(238, 238, 238))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 150, 136))
    palette.setColor(QPalette.Highlight, QColor(255, 87, 34))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette

THEMES = {
    "Dark": Theme(dark_palette, GLOBAL_STYLE_SHEET_DARK),
    "Green": Theme(green_palette, GLOBAL_STYLE_SHEET_GREEN),
    "light-blue": Theme(light_blue_palette, GLOBAL_STYLE_SHEET_LIGHT_BLUE),
    "Material": Theme(material_palette, GLOBAL_STYLE_SHEET_MATERIAL),
}

def apply_theme(app: QApplication, name: str):
    theme = THEMES.get(name)
    if not theme:
        return
    palette = theme.palette_func()
    app.setPalette(palette)
    app.setStyleSheet(theme.stylesheet)
