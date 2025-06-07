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

THEMES = {
    "Dark": Theme(dark_palette, GLOBAL_STYLE_SHEET_DARK),
    "Green": Theme(green_palette, GLOBAL_STYLE_SHEET_GREEN),
}

def apply_theme(app: QApplication, name: str):
    theme = THEMES.get(name)
    if not theme:
        return
    palette = theme.palette_func()
    app.setPalette(palette)
    app.setStyleSheet(theme.stylesheet)
