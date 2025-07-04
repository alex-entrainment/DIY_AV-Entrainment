import copy
from common.sequence_model import Step, Oscillator, StrobeSet, Waveform, PatternMode

class StepController:
    def __init__(self):
        self.steps = []

    def add_default_step(self, mode):
        if self.steps:
            prev = self.steps[-1]
            if mode == "Combined":
                po = prev.oscillators[0] if prev.oscillators else Oscillator(12.0, 12.0, Waveform.SQUARE)
                no = Oscillator(
                    start_freq=po.end_freq,
                    end_freq=po.end_freq,
                    waveform=po.waveform,
                    start_duty=po.end_duty,
                    end_duty=po.end_duty,
                    enable_rfm=po.enable_rfm,
                    rfm_range=po.rfm_range,
                    rfm_speed=po.rfm_speed,
                    phase_pattern=po.phase_pattern,
                    brightness_pattern=po.brightness_pattern,
                    pattern_strength=po.pattern_strength,
                    pattern_freq=po.pattern_freq
                )
                ns = StrobeSet(list(range(6)), po.end_duty, po.end_duty, [1.0])
                step = Step(30, "New Step", [no], [ns])
            elif mode == "Split":
                oscillators = []
                for i in range(2):
                    po = prev.oscillators[i] if i < len(prev.oscillators) else Oscillator(12.0, 12.0, Waveform.SQUARE)
                    no = Oscillator(
                        start_freq=po.end_freq,
                        end_freq=po.end_freq,
                        waveform=po.waveform,
                        start_duty=po.end_duty,
                        end_duty=po.end_duty,
                        enable_rfm=po.enable_rfm,
                        rfm_range=po.rfm_range,
                        rfm_speed=po.rfm_speed,
                        phase_pattern=po.phase_pattern,
                        brightness_pattern=po.brightness_pattern,
                        pattern_strength=po.pattern_strength,
                        pattern_freq=po.pattern_freq
                    )
                    oscillators.append(no)
                if len(prev.strobe_sets) >= 2:
                    ps1 = prev.strobe_sets[0]
                    ps2 = prev.strobe_sets[1]
                else:
                    ps1 = StrobeSet([0,2,4], 50, 50, [1.0, 0.0])
                    ps2 = StrobeSet([1,3,5], 50, 50, [0.0, 1.0])
                ns1 = StrobeSet([0,2,4], ps1.end_intensity, ps1.end_intensity, [1.0, 0.0])
                ns2 = StrobeSet([1,3,5], ps2.end_intensity, ps2.end_intensity, [0.0, 1.0])
                step = Step(30, "New Step", oscillators, [ns1, ns2])
            else:  # Independent
                oscillators = []
                strobe_sets = []
                for i in range(6):
                    po = prev.oscillators[i] if i < len(prev.oscillators) else Oscillator(12.0, 12.0, Waveform.SQUARE)
                    no = Oscillator(
                        start_freq=po.end_freq,
                        end_freq=po.end_freq,
                        waveform=po.waveform,
                        start_duty=po.end_duty,
                        end_duty=po.end_duty,
                        enable_rfm=po.enable_rfm,
                        rfm_range=po.rfm_range,
                        rfm_speed=po.rfm_speed,
                        phase_pattern=po.phase_pattern,
                        brightness_pattern=po.brightness_pattern,
                        pattern_strength=po.pattern_strength,
                        pattern_freq=po.pattern_freq
                    )
                    oscillators.append(no)
                    weights = [0.0] * 6
                    weights[i] = 1.0
                    if i < len(prev.strobe_sets):
                        ps = prev.strobe_sets[i]
                        ns = StrobeSet([i], ps.end_intensity, ps.end_intensity, weights)
                    else:
                        ns = StrobeSet([i], 50, 50, weights)
                    strobe_sets.append(ns)
                step = Step(30, "New Step", oscillators, strobe_sets)
        else:
            if mode == "Combined":
                step = Step(30, "New Step", [Oscillator(12.0, 12.0, Waveform.SQUARE)],
                            [StrobeSet(list(range(6)), 50, 50, [1.0])])
            elif mode == "Split":
                step = Step(30, "New Step", [Oscillator(12.0, 12.0, Waveform.SQUARE),
                                             Oscillator(12.0, 12.0, Waveform.SQUARE)],
                            [StrobeSet([0,2,4], 50, 50, [1.0, 0.0]),
                             StrobeSet([1,3,5], 50, 50, [0.0, 1.0])])
            else:  # Independent
                oscillators = []
                strobe_sets = []
                for i in range(6):
                    osc = Oscillator(12.0, 12.0, Waveform.SQUARE)
                    oscillators.append(osc)
                    weights = [0.0] * 6
                    weights[i] = 1.0
                    s = StrobeSet([i], 50, 50, weights)
                    strobe_sets.append(s)
                step = Step(30, "New Step", oscillators, strobe_sets)
        self.steps.append(step)
        return step

    def duplicate_step(self, index):
        if index < 0 or index >= len(self.steps):
            return None
        original = self.steps[index]
        new_step = copy.deepcopy(original)
        new_step.description += " (Copy)"
        self.steps.insert(index + 1, new_step)
        return new_step

    def remove_step(self, index):
        if 0 <= index < len(self.steps):
            return self.steps.pop(index)
        return None

    def move_step_up(self, index):
        if index > 0 and index < len(self.steps):
            self.steps[index], self.steps[index - 1] = self.steps[index - 1], self.steps[index]
            return True
        return False

    def move_step_down(self, index):
        if index >= 0 and index < len(self.steps) - 1:
            self.steps[index], self.steps[index + 1] = self.steps[index + 1], self.steps[index]
            return True
        return False

    def update_sequence_duration(self):
        total = sum(step.duration for step in self.steps)
        hrs = total // 3600
        mins = (total % 3600) // 60
        secs = total % 60
        return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
