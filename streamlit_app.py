"""
FraudShield AI - Root Streamlit Entry Point for Streamlit Community Cloud.
Automatically runs the dashboard application.
"""

import os
import sys
import runpy

# Ensure root directory and src directory are in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")

for path in [root_dir, src_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Execute main dashboard app
app_path = os.path.join(root_dir, "dashboard", "app.py")
runpy.run_path(app_path, run_name="__main__")
