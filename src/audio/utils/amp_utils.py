import math

MIN_DB = -60.0


def amplitude_to_db(amplitude: float) -> float:
    """Convert linear amplitude (0.0-1.0+) to dBFS."""
    if amplitude <= 0:
        return MIN_DB
    return 20.0 * math.log10(amplitude)


def db_to_amplitude(db: float) -> float:
    """Convert dBFS value to linear amplitude."""
    if db <= MIN_DB:
        return 0.0
    return 10 ** (db / 20.0)
