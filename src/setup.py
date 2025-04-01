# setup.py
import configparser
import platform
import sys
from pathlib import Path
import os

# --- Configuration ---
CONFIG_FILE = 'config.ini' # Name of the configuration file to create/update

# --- Helper Functions ---

def get_input_with_default(prompt, default_value):
    """Gets user input, showing the default value and handling empty input."""
    user_input = input(f"{prompt} [{default_value}]: ")
    return user_input.strip() or default_value

def validate_path(path_str, check_exists=True, is_dir=False, is_file=False, create_dir_if_not_exist=False):
    """
    Validates a path string. Checks existence and type (dir/file).
    Optionally creates directories. Returns validated Path object or None, plus error message.
    """
    if not path_str:
        return None, "Path cannot be empty."

    # Normalize path separators for the current OS
    try:
         # Handle potential invalid characters for the OS path format
         path_str_cleaned = path_str.replace('/', os.sep).replace('\\', os.sep)
         # Resolve relative paths (like '.') based on current working directory
         path = Path(path_str_cleaned).resolve()
    except Exception as e:
         return None, f"Invalid path format '{path_str}': {e}"

    if not check_exists:
         return path, None # Return path object without checking existence if requested

    if is_dir:
        if path.is_dir():
            return path, None
        elif create_dir_if_not_exist:
             try:
                  path.mkdir(parents=True, exist_ok=True)
                  print(f"Created directory: {path}")
                  return path, None
             except Exception as e:
                  return None, f"Could not create directory '{path}': {e}"
        else:
            return None, f"Directory does not exist: '{path}'"
    elif is_file:
        if path.is_file():
            return path, None
        else:
            # Provide more context if file missing
            if not path.parent.is_dir():
                 return None, f"Parent directory does not exist for file: '{path.parent}'"
            else:
                 return None, f"File does not exist: '{path}'"
    else: # Just check if path exists (can be file or dir)
        if path.exists():
            return path, None
        else:
            return None, f"Path does not exist: '{path}'"

# --- Main Setup Logic ---

def main():
    print("--- Project Configuration Setup ---")
    print(f"This script will configure settings for both the converter")
    print(f"(json_to_cpp_converter.py) and the controller (controller.py).")
    print(f"Settings are saved to '{CONFIG_FILE}' in the current directory.")

    config = configparser.ConfigParser()
    existing_config = {} # Stores loaded config as dict {section: {key: value}}
    config_path = Path(CONFIG_FILE)

    # Load existing config if it exists
    if config_path.exists():
        try:
            config.read(config_path)
            # Read existing values into the dictionary
            for section in config.sections():
                existing_config[section] = dict(config.items(section))

            print(f"\nLoaded existing configuration from {config_path}")
            reconfigure = input("Do you want to reconfigure all settings? (y/N): ").strip().lower()
            if reconfigure != 'y':
                print("Exiting setup without changes.")
                return
            else:
                # Clear config object to rebuild it, but keep existing_config for defaults
                config = configparser.ConfigParser()
        except Exception as e:
            print(f"Warning: Could not properly read existing {CONFIG_FILE}. Starting fresh. Error: {e}")
            config = configparser.ConfigParser() # Reset config object on error
            existing_config = {}

    # Ensure sections exist in the config object we're building
    if 'Paths' not in config: config.add_section('Paths')
    if 'PlatformIO' not in config: config.add_section('PlatformIO')
    if 'Controller' not in config: config.add_section('Controller')

    os_name = platform.system()
    print(f"\nDetected Operating System: {os_name}")
    is_windows = (os_name == 'Windows')
    is_linux = (os_name == 'Linux') # Assuming Pi runs Linux

    # === Serial Port (Required for Controller on Both Platforms) ===
    print("\n[Controller Configuration]")
    default_serial = ''
    # Get default from existing config OR provide OS-specific common defaults
    existing_controller_cfg = existing_config.get('Controller', {})
    if is_windows:
        default_serial = existing_controller_cfg.get('serial_port', 'COM3')
    elif is_linux:
        default_serial_options = ['/dev/ttyACM0', '/dev/ttyUSB0']
        default_serial = existing_controller_cfg.get('serial_port', default_serial_options[0])
        print(f"Common serial ports on Linux/Pi: {', '.join(default_serial_options)}")
    else: # Other OS (e.g., macOS)
        default_serial = existing_controller_cfg.get('serial_port', '/dev/tty.usbmodemXXXX')

    while True:
        serial_port = get_input_with_default("Enter Serial Port for ESP32 device", default_serial)
        if serial_port:
            config['Controller']['serial_port'] = serial_port
            break
        else:
            print("Serial port cannot be empty.")

    # === Windows-Specific Settings (for Converter/Uploader) ===
    if is_windows:
        print("\n[Converter/Uploader Configuration (Windows Only)]")
        existing_paths_cfg = existing_config.get('Paths', {})
        existing_pio_cfg = existing_config.get('PlatformIO', {})

        # --- PlatformIO Project Path ---
        default_pio_project = existing_paths_cfg.get('project_root', str(Path.home() / "Documents" / "PlatformIO" / "Projects" / "Your_Project_Name_Here"))
        print("\nEnter the full path to the root directory of your PlatformIO project.")
        while True:
            pio_project_str = get_input_with_default("PlatformIO Project Root Path", default_pio_project)
            valid_path, error = validate_path(pio_project_str, is_dir=True, check_exists=True)
            if valid_path:
                config['Paths']['project_root'] = str(valid_path) # Store validated, absolute path string
                break
            else:
                print(f"Validation Error: {error}")

        # --- Controller Script Path ---
        default_controller_script = existing_paths_cfg.get('controller_script', str(Path.cwd() / "controller.py")) # Default to current dir relative to setup.py
        print("\nEnter the full path to the 'controller.py' script (used by the converter to list sequences).")
        while True:
            controller_script_str = get_input_with_default("Path to controller.py", default_controller_script)
            valid_path, error = validate_path(controller_script_str, is_file=True, check_exists=True)
            if valid_path:
                config['Paths']['controller_script'] = str(valid_path)
                break
            else:
                print(f"Validation Error: {error}")

        # --- PlatformIO Executable Path ---
        default_pio_exe = existing_paths_cfg.get('pio_executable', str(Path.home() / ".platformio" / "penv" / "Scripts" / "platformio.exe"))
        print("\nEnter the full path to the 'platformio.exe' command-line tool.")
        while True:
            pio_exe_str = get_input_with_default("Path to platformio.exe", default_pio_exe)
            valid_path, error = validate_path(pio_exe_str, is_file=True, check_exists=True)
            if valid_path:
                config['Paths']['pio_executable'] = str(valid_path)
                break
            else:
                print(f"Validation Error: {error}")

        # --- PlatformIO Environment Name ---
        default_pio_env = existing_pio_cfg.get('pio_environment', 'seeed_xiao_esp32c3') # Your specific default
        print("\nEnter the name of the PlatformIO environment for your ESP32 board (from platformio.ini).")
        while True:
            pio_env_name = get_input_with_default("PlatformIO Environment Name", default_pio_env)
            if pio_env_name:
                config['PlatformIO']['pio_environment'] = pio_env_name
                break
            else:
                print("PlatformIO Environment name cannot be empty.")

    elif is_linux:
        print("\nSkipping Windows-specific PlatformIO path configuration for Linux/Pi.")
        # Set empty values for consistency if the converter script accidentally reads them on Linux
        config['Paths']['project_root'] = ''
        config['Paths']['controller_script'] = ''
        config['Paths']['pio_executable'] = ''
        config['PlatformIO']['pio_environment'] = ''
    else:
         print(f"\nSkipping Windows-specific PlatformIO path configuration for {os_name}.")
         config['Paths']['project_root'] = ''
         config['Paths']['controller_script'] = ''
         config['Paths']['pio_executable'] = ''
         config['PlatformIO']['pio_environment'] = ''


    # === Save Configuration ===
    try:
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        print(f"\nConfiguration saved successfully to '{config_path.resolve()}'") # Show absolute path
    except Exception as e:
        print(f"\nError saving configuration to {config_path}: {e}")

if __name__ == "__main__":
    main()
