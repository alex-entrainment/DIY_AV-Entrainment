# json_to_cpp_converter.py (Processes all JSON files in script directory)
import json
from pathlib import Path
import math # For rounding
import os
import re # For searching/updating files
import subprocess # For running PlatformIO
import configparser # For reading config
import sys # For exiting

# --- Function to convert JSON to C++ implementation string ---
# (convert_json_to_cpp function remains the same - generates non-static)
def convert_json_to_cpp(json_file_path, cpp_function_name):
    """
    Converts a sequence defined in a specific JSON format to a C++ function body string.
    (No longer uses 'static' for the generated function)
    """
    try:
        json_path = Path(json_file_path)
        if not json_path.is_file():
            print(f"Error: JSON file not found at {json_path}")
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'steps' not in data or not isinstance(data['steps'], list):
            print(f"Error: JSON file '{json_path.name}' must contain a 'steps' list.")
            return None

        json_steps = data['steps']
        if not json_steps:
             print(f"Warning: JSON file '{json_path.name}' 'steps' list is empty.")
             return f"\nvoid {cpp_function_name}() {{\n    Serial.println(\"Warning: Sequence {cpp_function_name} has no steps.\");\n}}\n"

        cpp_steps_code = []

        for i, step_data in enumerate(json_steps):
            required_keys = ['duration', 'oscillators', 'strobe_sets', 'description']
            if not all(key in step_data for key in required_keys):
                print(f"Error: Step {i} in '{json_path.name}' is missing required keys ({required_keys}).")
                return None
            if not isinstance(step_data['oscillators'], list) or len(step_data['oscillators']) < 1:
                print(f"Error: Step {i} in '{json_path.name}' must have at least 1 'oscillators' defined.")
                return None
            if not isinstance(step_data['strobe_sets'], list) or len(step_data['strobe_sets']) < 1:
                 print(f"Error: Step {i} in '{json_path.name}' must have at least 1 'strobe_sets' defined.")
                 return None

            try:
                duration_ms = int(step_data['duration'] * 1000)
                description = step_data.get('description', f'Step {i}')
                is_split_config = len(step_data['strobe_sets']) >= 2

                osc_a_index = 0
                strobe_a_index = 0
                start_freq_a = max(1, int(step_data['oscillators'][osc_a_index].get('start_freq', 0) * 100 + 0.5))
                end_freq_a = max(1, int(step_data['oscillators'][osc_a_index].get('end_freq', 0) * 100 + 0.5))
                waveform_a = int(step_data['oscillators'][osc_a_index].get('waveform', 2))
                start_duty_a = int(step_data['oscillators'][osc_a_index].get('start_duty', 50))
                end_duty_a = int(step_data['oscillators'][osc_a_index].get('end_duty', 50))
                start_brightness_a = int(step_data['strobe_sets'][strobe_a_index].get('start_intensity', 100))
                end_brightness_a = int(step_data['strobe_sets'][strobe_a_index].get('end_intensity', 100))

                if is_split_config:
                    osc_b_index = 1 if len(step_data['oscillators']) > 1 else 0
                    strobe_b_index = 1
                else:
                    osc_b_index = 0
                    strobe_b_index = 0

                if osc_b_index >= len(step_data['oscillators']):
                    print(f"Warning: Step {i} in '{json_path.name}' - Using oscillator[0] for Group B as oscillator[{osc_b_index}] doesn't exist.")
                    osc_b_index = 0
                if strobe_b_index >= len(step_data['strobe_sets']):
                    print(f"Warning: Step {i} in '{json_path.name}' - Using strobe_set[0] for Group B as strobe_set[{strobe_b_index}] doesn't exist.")
                    strobe_b_index = 0

                start_freq_b = max(1, int(step_data['oscillators'][osc_b_index].get('start_freq', 0) * 100 + 0.5))
                end_freq_b = max(1, int(step_data['oscillators'][osc_b_index].get('end_freq', 0) * 100 + 0.5))
                waveform_b = int(step_data['oscillators'][osc_b_index].get('waveform', 1))
                start_duty_b = int(step_data['oscillators'][osc_b_index].get('start_duty', 50))
                end_duty_b = int(step_data['oscillators'][osc_b_index].get('end_duty', 50))
                start_brightness_b = int(step_data['strobe_sets'][strobe_b_index].get('start_intensity', 100))
                end_brightness_b = int(step_data['strobe_sets'][strobe_b_index].get('end_intensity', 100))

                step_comment = f"// Step {i}: {description}"
                step_code = f"""\
    {{ .durationMs = {duration_ms},
      .startFreqA = {start_freq_a}, .endFreqA = {end_freq_a}, .startDutyA = {start_duty_a}, .endDutyA = {end_duty_a}, .startBrightnessA = {start_brightness_a}, .endBrightnessA = {end_brightness_a}, .waveformA = {waveform_a},
      .startFreqB = {start_freq_b}, .endFreqB = {end_freq_b}, .startDutyB = {start_duty_b}, .endDutyB = {end_duty_b}, .startBrightnessB = {start_brightness_b}, .endBrightnessB = {end_brightness_b}, .waveformB = {waveform_b}
    }}"""
                cpp_steps_code.append(step_comment)
                cpp_steps_code.append(step_code + ",")

            except KeyError as e:
                print(f"Error: Missing key {e} in step {i} of '{json_path.name}'. Check JSON structure.")
                return None
            except (TypeError, ValueError, IndexError) as e:
                print(f"Error processing data type/index in step {i} of '{json_path.name}': {e}")
                return None
        # --- End Step Loop ---

        if cpp_steps_code[-1].endswith(","):
           cpp_steps_code[-1] = cpp_steps_code[-1][:-1] # Remove trailing comma

        steps_initializer_code = "\n".join(cpp_steps_code)

        cpp_output = f"""\

// AUTOGENERATED FUNCTION - DO NOT EDIT MANUALLY within this block
// Generated from: {json_path.name}
// Function name: {cpp_function_name}
// -------------------------------------------------------------
void {cpp_function_name}() {{
    SequenceStep steps[] = {{
{steps_initializer_code}
    }};
    constexpr int stepCount = sizeof(steps) / sizeof(steps[0]);
    Serial.print("Running sequence: {cpp_function_name} ("); Serial.print(stepCount); Serial.println(" steps)");

    // Requires declaration/definition for: isSequenceRunning, runSmoothSequence
    isSequenceRunning = true;
    runSmoothSequence(steps, stepCount);
    isSequenceRunning = false;
}}
// -------------------------------------------------------------
// END AUTOGENERATED FUNCTION - {cpp_function_name}
// -------------------------------------------------------------
"""
        return cpp_output

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file '{json_file_path}' - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during conversion of '{json_file_path.name}': {e}")
        return None

# --- Function to append content to a file ---
def append_to_file(file_path, content_to_append):
    """Appends content to a file, ensuring a newline before the appended text if needed."""
    needs_newline = False
    if file_path.exists() and file_path.stat().st_size > 0:
        try:
            with open(file_path, 'rb') as f:
                f.seek(-1, os.SEEK_END); last_byte = f.read(1)
                if last_byte not in (b'\n', b'\r'): needs_newline = True
                elif last_byte == b'\r':
                     f.seek(-2, os.SEEK_END);
                     if f.read(1) != b'\n': needs_newline = True
        except OSError: pass
        except Exception as e: needs_newline = True # Assume needed if check fails

    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            if needs_newline: f.write('\n')
            f.write(content_to_append)
            if not content_to_append.endswith(('\n', '\r')): f.write('\n')
        # print(f"Successfully appended content to '{file_path.name}'") # Make less verbose
        return True
    except Exception as e:
        print(f"Error appending to file '{file_path}': {e}")
        return False

# --- Function to update main.cpp ---
def update_main_cpp(main_cpp_path, cpp_function_name):
    """
    Updates main.cpp to add the new sequence name to setup() and loop().
    Avoids adding duplicates. Returns True if changes were made, False if no changes needed/made, None on error.
    """
    if not main_cpp_path.is_file(): return None # Error: file not found

    try:
        with open(main_cpp_path, 'r', encoding='utf-8') as f: lines = f.readlines()

        content = "".join(lines)
        setup_line_pattern = re.compile(r'Serial\.println\("Known sequence names:.*\b' + re.escape(cpp_function_name) + r'\b.*"\);')
        loop_line_pattern = re.compile(r'else\s+if\s+\(seqName\s*==\s*"' + re.escape(cpp_function_name) + r'"\)\s*{\s*' + re.escape(cpp_function_name) + r'\(\);\s*\}', re.MULTILINE)
        setup_already_updated = bool(setup_line_pattern.search(content))
        loop_already_updated = bool(loop_line_pattern.search(content))

        if setup_already_updated and loop_already_updated: return False # No changes needed

        setup_insert_line_index = -1; known_names_marker = 'Serial.println("Known sequence names:'
        loop_insert_line_index = -1; unknown_name_marker = 'Serial.println("Error: Unknown sequence name.");'

        for i, line in enumerate(lines):
            if setup_insert_line_index == -1 and known_names_marker in line: setup_insert_line_index = i
            if unknown_name_marker in line: loop_insert_line_index = i # Find error line first

        # Find the 'else {' just before the error line for loop insertion
        if loop_insert_line_index != -1:
            found_else = False
            for i in range(loop_insert_line_index - 1, max(0, loop_insert_line_index - 5), -1):
                if lines[i].strip() in ('} else {', 'else {'): loop_insert_line_index = i; found_else = True; break
            if not found_else: loop_insert_line_index = -1 # Reset if suitable 'else' wasn't found

        made_changes = False
        # Update setup()
        if setup_insert_line_index != -1 and not setup_already_updated:
            target_line = lines[setup_insert_line_index]; insert_pos = target_line.rfind('");')
            if insert_pos != -1:
                 text_before_insert = target_line[:insert_pos].rstrip(); separator = ", " if text_before_insert[-1] != '(' else ""
                 new_line = text_before_insert + separator + cpp_function_name + target_line[insert_pos:]
                 lines[setup_insert_line_index] = new_line; made_changes = True
                 print(f"  - Updated setup() known names for '{cpp_function_name}'.")
            else: print(f"  - Warning: Could not find marker in setup() line for '{cpp_function_name}'.")

        # Update loop()
        if loop_insert_line_index != -1 and not loop_already_updated:
            indentation = re.match(r"^(\s*)", lines[loop_insert_line_index]).group(1) if re.match(r"^(\s*)", lines[loop_insert_line_index]) else ""
            new_else_if_block = [ f"{indentation}}} else if (seqName == \"{cpp_function_name}\") {{\n",
                                  f"{indentation}    {cpp_function_name}(); // Autogenerated sequence call\n" ]
            lines[loop_insert_line_index:loop_insert_line_index] = new_else_if_block; made_changes = True
            print(f"  - Added 'else if' block for '{cpp_function_name}' to loop().")
        elif loop_insert_line_index == -1:
             print(f"  - Error: Could not find insertion point in loop() for '{cpp_function_name}'.")
             return None # Critical error if loop can't be updated

        # Write back if changes were made
        if made_changes:
            print(f"  - Writing changes to '{main_cpp_path.name}'.")
            with open(main_cpp_path, 'w', encoding='utf-8') as f: f.writelines(lines)
            return True
        else: return False # No changes needed/made

    except Exception as e:
        print(f"Error updating '{main_cpp_path.name}': {e}")
        return None # Error

# --- Function to update controller.py ---
def update_controller_py(controller_py_path, cpp_function_name):
    """
    Updates controller.py to add the new sequence name to its list/printout.
    Avoids adding duplicates. Returns True if changes were made, False if no changes needed/made, None on error.
    """
    if not controller_py_path.is_file(): return None # Error: file not found

    try:
        with open(controller_py_path, 'r', encoding='utf-8') as f: lines = f.readlines()

        target_print_pattern = re.compile(r'print\(.*["\']\s*-\s*' + re.escape(cpp_function_name) + r'\s*["\']\s*.*\)\s*#?.*')
        already_present = any(target_print_pattern.search(line) for line in lines)
        if already_present: return False # No changes needed

        insert_marker_options = [ 'print("Enter commands:")', 'print("\\nEnter commands' ]
        insert_line_index = -1; start_marker = "Known available sequences"

        found_start_marker = False
        for i, line in enumerate(lines):
             # Look for the line AFTER the known sequences intro
             if start_marker in line: found_start_marker = True
             # Find the command prompt line, but only AFTER the start marker
             if found_start_marker and any(marker in line.strip() for marker in insert_marker_options):
                  insert_line_index = i; break

        if insert_line_index == -1:
            print(f"  - Error: Could not find insertion point markers (around '{start_marker}' / '{insert_marker_options[0]}') in '{controller_py_path.name}'.")
            return None

        # Determine indentation
        indentation = "    "
        if insert_line_index > 0:
             match = re.match(r"^(\s*)", lines[insert_line_index - 1])
             if match: indentation = match.group(1)

        new_line_to_insert = f'{indentation}print("  - {cpp_function_name}") # Added by converter\n'
        lines.insert(insert_line_index, new_line_to_insert)
        print(f"  - Added known sequence name '{cpp_function_name}' to '{controller_py_path.name}'.")

        # Write back
        print(f"  - Writing changes to '{controller_py_path.name}'.")
        with open(controller_py_path, 'w', encoding='utf-8') as f: f.writelines(lines)
        return True # Changes made

    except Exception as e:
        print(f"Error updating '{controller_py_path.name}': {e}")
        return None # Error

# --- Function to Run PlatformIO Upload ---
def run_pio_upload(project_path, pio_executable, environment):
    """Runs the PlatformIO upload command."""
    if not project_path.is_dir(): return False
    pio_exe_path = Path(pio_executable)
    if not pio_exe_path.is_file(): return False

    command = [ str(pio_exe_path), "run", "--target", "upload", "--environment", environment ]
    print(f"\nAttempting PlatformIO Upload...")
    print(f"Project: {project_path}")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run( command, cwd=project_path, capture_output=True,
                                text=True, check=False, encoding='utf-8')
        print("\n--- PlatformIO Output ---")
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"--- PlatformIO Errors/Warnings ---\n{result.stderr}\n--------------------------")
        if result.returncode == 0: print("\nPlatformIO upload completed successfully."); return True
        else: print(f"\nPlatformIO upload failed (exit code {result.returncode})."); return False
    except FileNotFoundError: print(f"FATAL ERROR: PlatformIO executable not found at '{command[0]}'."); return False
    except Exception as e: print(f"Error during PlatformIO upload: {e}"); return False

# --- Main Execution Logic ---
if __name__ == "__main__":

    # --- Configuration Loading ---
    config = configparser.ConfigParser()
    script_dir = Path(__file__).parent
    config_file = script_dir / r"C:/Users/alexb/DIY_AV-Entrainment/src/config.ini"

    if not config_file.is_file():
        print(f"ERROR: Configuration file '{config_file}' not found.")
        print(f"Please run setup.py in the script directory ({script_dir}) first.")
        sys.exit(1)

    try:
        config.read(config_file)
        project_base_path_str = config.get('Paths', 'project_root', fallback='')
        controller_py_path_str = config.get('Paths', 'controller_script', fallback='')
        pio_executable_path_str = config.get('Paths', 'pio_executable', fallback='')
        pio_environment_name = config.get('PlatformIO', 'pio_environment', fallback='')

        if not all([project_base_path_str, controller_py_path_str, pio_executable_path_str, pio_environment_name]):
             print("ERROR: Required paths/settings missing in config.ini. Run setup.py on dev machine.")
             sys.exit(1)

        project_base_path = Path(project_base_path_str.replace('/', os.sep).replace('\\', os.sep))
        controller_py_file = Path(controller_py_path_str.replace('/', os.sep).replace('\\', os.sep))
        pio_executable_path = Path(pio_executable_path_str.replace('/', os.sep).replace('\\', os.sep))

        src_path = project_base_path / "src"
        cpp_target_file = src_path / "sequences.cpp"
        hpp_target_file = src_path / "sequences.hpp"
        main_cpp_file = src_path / "main.cpp"

        # --- Path Existence Checks ---
        critical_paths_ok = True
        paths_to_check = {
            "Project Base Path": (project_base_path, True), # is_dir
            "Controller Script": (controller_py_file, False), # is_file
            "PlatformIO Executable": (pio_executable_path, False), # is_file
            "Project Source Dir": (src_path, True), # is_dir
            "main.cpp": (main_cpp_file, False), # is_file
            # Target files sequences.cpp/hpp might not exist yet, append handles creation
        }
        for name, (path_obj, is_dir_check) in paths_to_check.items():
             exists = path_obj.is_dir() if is_dir_check else path_obj.is_file()
             if not exists:
                  print(f"FATAL ERROR: Required path/file not found: '{path_obj}' ({name})")
                  critical_paths_ok = False

        if not critical_paths_ok:
             print("Aborting due to missing critical paths configured in config.ini or project structure.")
             sys.exit(1)

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"ERROR: Missing configuration section/key in '{config_file}': {e}. Run setup.py."); sys.exit(1)
    except Exception as e:
        print(f"ERROR reading configuration '{config_file}': {e}"); sys.exit(1)

    # --- Find and Process JSON Files ---
    json_files = list(script_dir.glob('*.json'))
    if not json_files:
        print(f"No .json files found in the script directory: {script_dir}")
        sys.exit(0)

    print(f"\nFound {len(json_files)} JSON file(s) to process in {script_dir}")

    processed_files = []
    failed_files = []
    any_changes_made = False # Track if any file was actually modified

    for json_path in json_files:
        print(f"\n--- Processing: {json_path.name} ---")
        cpp_function_name = json_path.stem
        # Sanitize name
        if not cpp_function_name.isidentifier():
            sanitized_name = ''.join(c if c.isalnum() else '_' for c in cpp_function_name)
            if not sanitized_name or not (sanitized_name[0].isalpha() or sanitized_name[0] == '_'): sanitized_name = "_" + sanitized_name
            cpp_keywords = {"for", "while", "if", "else", "switch", "case", "int", "void", "static", "class", "struct", "bool", "true", "false"}
            if sanitized_name in cpp_keywords: sanitized_name += "_seq"
            if sanitized_name != cpp_function_name:
                 print(f"  Using sanitized function name: '{sanitized_name}'")
            cpp_function_name = sanitized_name

        # --- Convert ---
        generated_cpp_code = convert_json_to_cpp(json_path, cpp_function_name)
        if not generated_cpp_code:
            failed_files.append(json_path.name)
            continue # Skip to next file

        # --- Check Duplicates ---
        impl_exists = False; decl_exists = False
        try:
            if cpp_target_file.exists():
                with open(cpp_target_file, 'r', encoding='utf-8') as f: content = f.read()
                if f"\nvoid {cpp_function_name}()" in content and f"// Function name: {cpp_function_name}" in content: impl_exists = True
            if hpp_target_file.exists():
                with open(hpp_target_file, 'r', encoding='utf-8') as f: content = f.read()
                if f"void {cpp_function_name}();" in content: decl_exists = True
        except Exception as e: print(f"Warning: Error checking duplicates for {cpp_function_name}: {e}")

        if impl_exists: print(f"  - Implementation for '{cpp_function_name}' seems to exist in '{cpp_target_file.name}'.")
        if decl_exists: print(f"  - Declaration for '{cpp_function_name}' seems to exist in '{hpp_target_file.name}'.")

        # --- Append/Update C++ Files ---
        cpp_declaration = f"void {cpp_function_name}(); // Generated from {json_path.name}"
        impl_ok = impl_exists or append_to_file(cpp_target_file, generated_cpp_code)
        decl_ok = False
        if impl_ok: decl_ok = decl_exists or append_to_file(hpp_target_file, cpp_declaration)

        # --- Update main.cpp ---
        main_cpp_result = None # None=Error, False=NoChange, True=Changed
        if impl_ok and decl_ok:
             main_cpp_result = update_main_cpp(main_cpp_file, cpp_function_name)
             if main_cpp_result is None: # Critical error updating main.cpp
                  print(f"  - CRITICAL ERROR updating main.cpp for {cpp_function_name}. Check file content/markers.")
                  failed_files.append(f"{json_path.name} (main.cpp update error)")
                  continue # Skip rest for this file

        # --- Update controller.py ---
        controller_py_result = None
        if impl_ok and decl_ok and main_cpp_result is not None: # Proceed if C++ side seems consistent
             if controller_py_file.is_file():
                  controller_py_result = update_controller_py(controller_py_file, cpp_function_name)
                  if controller_py_result is None: # Error updating controller.py
                       print(f"  - ERROR updating controller.py for {cpp_function_name}.")
                       failed_files.append(f"{json_path.name} (controller.py update error)")
                       # Don't necessarily stop processing other files, but note failure
             else: print(f"  - Skipping controller.py update (file not found).")
        else: print(f"  - Skipping controller.py update due to issues with C++ file updates.")

        # Track if any actual file writes happened
        file_was_modified = (not impl_exists and impl_ok) or \
                            (not decl_exists and decl_ok) or \
                            (main_cpp_result == True) or \
                            (controller_py_result == True)
        if file_was_modified:
             any_changes_made = True

        processed_files.append(json_path.name)
        print(f"--- Finished Processing: {json_path.name} ---")
    # --- End JSON File Loop ---

    # --- Summary ---
    print("\n--- Processing Summary ---")
    if processed_files:
        print(f"Successfully processed (or found existing entries for):")
        for f in processed_files: print(f"  - {f}")
    if failed_files:
        print(f"\nFailed to fully process:")
        for f in failed_files: print(f"  - {f}")
    if not processed_files and not failed_files:
        print("No JSON files were found or processed.")

    # --- PlatformIO Upload ---
    if processed_files and not failed_files and any_changes_made:
        print("\nFile updates completed successfully.")
        upload_confirm = input("Do you want to attempt PlatformIO upload now? (Y/n): ").strip().lower()
        if upload_confirm == "" or upload_confirm == "y":
             run_pio_upload(project_base_path, pio_executable_path, pio_environment_name)
        else:
             print("Skipping PlatformIO upload.")
    elif not any_changes_made and not failed_files:
        print("\nNo changes were made to project files. Skipping PlatformIO upload.")
    elif failed_files:
        print("\nSkipping PlatformIO upload due to errors during processing.")
    else: # No JSON files found case
        print("\nNo files processed. Skipping PlatformIO upload.")


 
