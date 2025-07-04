import configparser
from pathlib import Path

CONFIG_FILE = Path("audio_config.ini")

def main():
    config = configparser.ConfigParser()
    if not CONFIG_FILE.exists():
        config["Audio"] = {"output_dir": "output"}
        with open(CONFIG_FILE, "w") as f:
            config.write(f)
        print(f"Created default {CONFIG_FILE}")
    else:
        config.read(CONFIG_FILE)
        print(f"Config loaded from {CONFIG_FILE}")

if __name__ == "__main__":
    main()
