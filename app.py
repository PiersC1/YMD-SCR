"""
Root entrypoint for Streamlit Cloud deployment.
Delegates to src.app.main()
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.app import main

if __name__ == "__main__":
    main()
