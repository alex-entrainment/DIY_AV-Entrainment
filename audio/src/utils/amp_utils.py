import math
import re

MIN_DB = -60.0
_AMP_PARAM_RE = re.compile(r'(?:^|_)(?:amp|amplitude|gain|level)(?:$|_)')


def amplitude_to_db(amplitude: float) -> float:
    """Convert linear amplitude (0.0-1.0+) to dBFS, clamped to MIN_DB."""
    if amplitude <= 0:
        return MIN_DB
    db = 20.0 * math.log10(amplitude)
    return db if db > MIN_DB else MIN_DB


def db_to_amplitude(db: float) -> float:
    """Convert dBFS value to linear amplitude."""
    if db <= MIN_DB:
        return 0.0
    return 10 ** (db / 20.0)


def is_amp_key(name: str) -> bool:
    """Return True if the parameter name refers to an amplitude value."""
    return bool(_AMP_PARAM_RE.search(name.lower()))
