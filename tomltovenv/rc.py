import os

# fmt: off

### Paths
PROJECT_ROOT = ".."  # !! Since we execute main.py from tomltovenv/, we need to go up one level
VENV_NAME = "venv"

### Flags
#' Re-solve environment and (over)write requirements.txt
REDO_REQUIREMENTS_TXT = False  
#' Install editable pip install . -e ?
EDITABLE = True  
#' Install pip install .[dev] ?
INSTALL_DEVTOOLS = True  


