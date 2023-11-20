"""Script to prepare the global python environment so that tomltovenv 
can run
- Installs packages required for tomltovenv
"""


import subprocess
import shlex

# import os
# import argparse
import sys
from pathlib import PurePath


def print_header(s: str) -> None:
    """Print a header with a string in the middle Adds two = before the
    string and then adds = after the string until total length of string
    is 80 reached
    """
    print("\n")  # ' two empty lines
    print("=" * 2, s, "=" * (80 - len(s) - 2))
    print()



### Get Python Executable that called this script
#' main() calls this script using different python executables
python = sys.executable

### Print header
#' Split path into parts and get last 4, then join them
python_s = PurePath(python).parts[-4:]
python_s = "/".join(python_s)
print_header(f"Preparing Environment: {python_s}")


# == Install packages required for tomltovenv ==================================

PREPARE_PACKAGES = [
    "toml",  #' to read the python version from pyproject.toml
    "icecream",
]


def import_or_install(
    packages: list,
    python: str = None,
) -> None:
    """Import packages, if not installed, install them"""
    python = "python" if python is None else python

    uninstalled = []
    for package in packages:
        try:
            __import__(package)  #' Dunder imports from string
            # print(f">>> __import__({package})")
            print(f"Package Importable: {package}")
        except ImportError:
            uninstalled.append(package)

    if uninstalled:
        packages_str = " ".join(packages)
        cmd = f"{python} -m pip install {packages_str}"
        print(f"\n$ {cmd}")
        cmd = shlex.split(cmd)
        subprocess.run(cmd)


### Install packages
import_or_install(packages=PREPARE_PACKAGES, python=python)
