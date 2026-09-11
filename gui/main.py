
# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')  # Streamlit renders images, not desktop Tk windows.

from gui.app import run_app


if __name__ == "__main__":
    run_app()
