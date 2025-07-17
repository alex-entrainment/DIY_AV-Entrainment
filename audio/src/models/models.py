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

    def _format_number(self, value):
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)

    def _get_beat_frequency(self, params, is_transition):
        """Return formatted beat frequency for display."""
        beat_keys = [k for k in params if "beat" in k.lower() and "freq" in k.lower()]

        start_keys = [k for k in beat_keys if k.lower().startswith("start")]
        end_keys = [k for k in beat_keys if k.lower().startswith("end")]
        normal_keys = [
            k
            for k in beat_keys
            if not k.lower().startswith(("start", "end", "target"))
        ]

        if is_transition:
            if start_keys and end_keys:
                s_val = params.get(start_keys[0])
                e_val = params.get(end_keys[0])
                try:
                    s_f = float(s_val)
                    e_f = float(e_val)
                    if abs(s_f - e_f) < 1e-6:
                        return f"{s_f:.2f}"
                    return f"{s_f:.2f}->{e_f:.2f}"
                except (ValueError, TypeError):
                    if s_val == e_val:
                        return str(s_val)
                    return f"{s_val}->{e_val}"
            if start_keys:
                return self._format_number(params.get(start_keys[0]))
            if end_keys:
                return self._format_number(params.get(end_keys[0]))

        if normal_keys:
            return self._format_number(params.get(normal_keys[0]))

        return "N/A"

    def refresh(self, steps=None):
        if steps is not None:
            self.steps = steps
        self.beginResetModel()
        self.endResetModel()


class VoiceModel(QAbstractTableModel):
    """Model holding a list of voice dictionaries for a selected step."""
    headers = [
        "Synth Function",
        "Carrier Freq",
        "Beat Freq",
        "Transition?",
        "Init Offset",
        "Post Offset",
        "Description",
    ]

    def __init__(self, voices=None, parent=None):
        super().__init__(parent)
        self.voices = voices if voices is not None else []

    def rowCount(self, parent=QModelIndex()):
        return len(self.voices)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

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
                beat_val = self._get_beat_frequency(params, is_transition)
                return beat_val
            if index.column() == 3:
                return "Yes" if is_transition else "No"
            if index.column() == 4:
                return self._format_number(params.get("initial_offset", 0.0)) if is_transition else "N/A"
            if index.column() == 5:
                return self._format_number(params.get("post_offset", 0.0)) if is_transition else "N/A"
            if index.column() == 6:
                return description
        return None

    def _format_number(self, value):
        """Return value formatted to two decimals if numeric."""
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)

    def _get_beat_frequency(self, params, is_transition):
        """Return formatted beat frequency for display."""
        beat_keys = [k for k in params if "beat" in k.lower() and "freq" in k.lower()]

        start_keys = [k for k in beat_keys if k.lower().startswith("start")]
        end_keys = [k for k in beat_keys if k.lower().startswith("end")]
        normal_keys = [
            k
            for k in beat_keys
            if not k.lower().startswith(("start", "end", "target"))
        ]

        if is_transition:
            if start_keys and end_keys:
                s_val = params.get(start_keys[0])
                e_val = params.get(end_keys[0])
                try:
                    s_f = float(s_val)
                    e_f = float(e_val)
                    if abs(s_f - e_f) < 1e-6:
                        return f"{s_f:.2f}"
                    return f"{s_f:.2f}->{e_f:.2f}"
                except (ValueError, TypeError):
                    if s_val == e_val:
                        return str(s_val)
                    return f"{s_val}->{e_val}"
            if start_keys:
                return self._format_number(params.get(start_keys[0]))
            if end_keys:
                return self._format_number(params.get(end_keys[0]))

        if normal_keys:
            return self._format_number(params.get(normal_keys[0]))

        return "N/A"

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False
        voice = self.voices[index.row()]
        if index.column() == 4:
            try:
                voice.setdefault('params', {})['initial_offset'] = float(value)
            except (ValueError, TypeError):
                return False
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        if index.column() == 5:
            try:
                voice.setdefault('params', {})['post_offset'] = float(value)
            except (ValueError, TypeError):
                return False
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        if index.column() == 6:
            voice['description'] = str(value).strip()
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() in (4, 5, 6):
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
