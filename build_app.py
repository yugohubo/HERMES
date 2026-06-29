import os
import subprocess
import sys

def build():
    print("=== HERMES PYINSTALLER BUILDER ===")
    
    # Check if pyinstaller is installed
    try:
        import PyInstaller
        print("PyInstaller is installed.")
    except ImportError:
        print("Error: PyInstaller is not installed in this Python environment.")
        print("Please run: pip install pyinstaller")
        sys.exit(1)

    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    entry_point = os.path.join(base_dir, "main_gui.py")

    if not os.path.exists(entry_point):
        print(f"Error: Entry point {entry_point} not found!")
        sys.exit(1)

    # Construct the PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",              # Create a single executable
        "--noconsole",            # Hide the console window
        "--name=HERMES",          # Name of the executable
        # Include custom imports directories if needed (PyInstaller usually auto-detects them)
        f"--paths={base_dir}",
        entry_point
    ]

    print("\nRunning PyInstaller build command:")
    print(" ".join(cmd))
    
    try:
        subprocess.check_call(cmd, cwd=base_dir)
        print("\n=== BUILD COMPLETED SUCCESSFULLY! ===")
        print("The standalone executable is located in: dist/HERMES.exe")
    except subprocess.CalledProcessError as e:
        print(f"\nError: PyInstaller failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    build()
