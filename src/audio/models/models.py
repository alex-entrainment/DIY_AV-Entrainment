from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex

class StepModel(QAbstractTableModel):
    """Model holding a list of step dictionaries."""
    headers = ["Duration (s)", "Description", "# Voices"]

    def __init__(self, steps=None, parent=None):
        super().__init__(parent)
        self.steps = steps if steps is not None else []

    def rowCount(self, parent=QModelIndex()):
        return len(self.steps)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        step = self.steps[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if index.column() == 0:
                return f"{step.get('duration', 0.0):.2f}"
            if index.column() == 1:
                return step.get('description', '')
            if index.column() == 2:
                return str(len(step.get('voices', [])))
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False
        step = self.steps[index.row()]
        if index.column() == 1:
            step['description'] = str(value).strip()
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() == 1:
            flags |= Qt.ItemIsEditable
        return flags

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section < len(self.headers):
                return self.headers[section]
        return super().headerData(section, orientation, role)

    def refresh(self, steps=None):
        if steps is not None:
            self.steps = steps
        self.beginResetModel()
        self.endResetModel()


class VoiceModel(QAbstractTableModel):
    """Model holding a list of voice dictionaries for a selected step."""
    headers = ["Synth Function", "Carrier Freq", "Transition?", "Description"]

    def __init__(self, voices=None, parent=None):
        super().__init__(parent)
        self.voices = voices if voices is not None else []

    def rowCount(self, parent=QModelIndex()):
        return len(self.voices)

    def columnCount(self, parent=QModelIndex()):
        return 4

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        voice = self.voices[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            func_name = voice.get('synth_function_name', 'N/A')
            params = voice.get('params', {})
            is_transition = voice.get('is_transition', False)
            description = voice.get('description', '')
            if index.column() == 0:
                return func_name
            if index.column() == 1:
                carrier = None
                if 'baseFreq' in params:
                    carrier = params['baseFreq']
                elif 'frequency' in params:
                    carrier = params['frequency']
                elif 'carrierFreq' in params:
                    carrier = params['carrierFreq']
                else:
                    freq_keys = [k for k in params if ('Freq' in k or 'Frequency' in k) and not k.startswith(('start','end','target'))]
                    if is_transition:
                        freq_keys = [k for k in params if k.startswith('start') and ('Freq' in k or 'Frequency' in k)] or freq_keys
                    carrier = params.get(freq_keys[0]) if freq_keys else 'N/A'
                try:
                    if carrier is not None and carrier != 'N/A':
                        return f"{float(carrier):.2f}"
                    return str(carrier)
                except (ValueError, TypeError):
                    return str(carrier)
            if index.column() == 2:
                return "Yes" if is_transition else "No"
            if index.column() == 3:
                return description
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False
        voice = self.voices[index.row()]
        if index.column() == 3:
            voice['description'] = str(value).strip()
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() == 3:
            flags |= Qt.ItemIsEditable
        return flags

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section < len(self.headers):
                return self.headers[section]
        return super().headerData(section, orientation, role)

    def refresh(self, voices=None):
        if voices is not None:
            self.voices = voices
        self.beginResetModel()
        self.endResetModel()
