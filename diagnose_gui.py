import sys
from PyQt6.QtWidgets import QApplication
from main_gui import HermesMainWindow

print("Starting diagnosis...")
app = QApplication(sys.argv)
print("QApplication created.")

class DiagnosticWindow(HermesMainWindow):
    def __init__(self):
        print("Calling super().__init__()...")
        super().__init__()
        print("super().__init__() complete.")

print("Instantiating DiagnosticWindow...")
try:
    w = DiagnosticWindow()
    print("Instance created successfully.")
except Exception as e:
    print(f"Python exception caught: {e}")
    import traceback
    traceback.print_exc()
