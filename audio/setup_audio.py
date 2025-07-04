import shutil
from pathlib import Path

CONFIG_SRC = Path(__file__).parent / 'src' / 'audio' / 'realtime_backend' / 'config.toml'
CONFIG_DIR = Path.home() / '.diy_av_audio'
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_DEST = CONFIG_DIR / 'config.toml'

if CONFIG_SRC.exists():
    shutil.copy(CONFIG_SRC, CONFIG_DEST)
    print(f"Copied {CONFIG_SRC} to {CONFIG_DEST}")
else:
    print(f"Source config not found: {CONFIG_SRC}")
