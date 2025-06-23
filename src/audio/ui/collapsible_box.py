from PyQt5.QtWidgets import QWidget, QToolButton, QVBoxLayout
from PyQt5.QtCore import Qt

class CollapsibleBox(QWidget):
    """A simple collapsible container with a toggle button."""

    def __init__(self, title: str = "", parent: QWidget = None):
        super().__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=True)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.clicked.connect(self._on_toggled)

        self.content_area = QWidget()

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addWidget(self.toggle_button)
        self._layout.addWidget(self.content_area)

    def _on_toggled(self, checked: bool):
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content_area.setVisible(checked)

    def setContentLayout(self, layout: QVBoxLayout):
        """Set the layout that holds the collapsible content."""
        self.content_area.setLayout(layout)
