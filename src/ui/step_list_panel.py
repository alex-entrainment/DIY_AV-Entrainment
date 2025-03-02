from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QListWidgetItem
from PyQt5.QtCore import pyqtSignal

class StepListPanel(QWidget):
    # Signals for actions and list changes.
    addStepClicked = pyqtSignal()
    duplicateStepClicked = pyqtSignal()
    removeStepClicked = pyqtSignal()
    moveUpClicked = pyqtSignal()
    moveDownClicked = pyqtSignal()
    stepSelectionChanged = pyqtSignal(int)  # current row index

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.step_list = QListWidget()
        self.step_list.setStyleSheet("""
            QListWidget::item {
                border: 1px solid gray;
                margin: 1px;
                padding: 3px;
            }
            QListWidget::item:selected {
                background-color: #b0d4f1;
            }
        """)
        layout.addWidget(self.step_list)

        # Create control buttons.
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Step")
        self.btn_duplicate = QPushButton("Duplicate Step")
        self.btn_remove = QPushButton("Remove Step")
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_down = QPushButton("Move Down")
        for btn in [self.btn_add, self.btn_duplicate, self.btn_remove, self.btn_move_up, self.btn_move_down]:
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        # Connect UI signals to our custom signals.
        self.btn_add.clicked.connect(self.addStepClicked)
        self.btn_duplicate.clicked.connect(self.duplicateStepClicked)
        self.btn_remove.clicked.connect(self.removeStepClicked)
        self.btn_move_up.clicked.connect(self.moveUpClicked)
        self.btn_move_down.clicked.connect(self.moveDownClicked)
        self.step_list.currentRowChanged.connect(self.stepSelectionChanged.emit)

    def add_step_item(self, description):
        item = QListWidgetItem(description)
        self.step_list.addItem(item)
        return item

    def insert_step_item(self, index, description):
        item = QListWidgetItem(description)
        self.step_list.insertItem(index, item)
        return item

    def update_step_item(self, index, description):
        if 0 <= index < self.step_list.count():
            self.step_list.item(index).setText(description)

    def remove_current_item(self):
        index = self.step_list.currentRow()
        if index >= 0:
            self.step_list.takeItem(index)

    def clear(self):
        self.step_list.clear()

    def set_current_row(self, row):
        self.step_list.setCurrentRow(row)
