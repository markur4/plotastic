"""Utilities for environment setup
1. Install a Python globally for your OS with correct version
2. Use global Python to call `venv.create` and make a virtual environment for
   each project 
3. Switch Jupyter Kernel to venv
4. Install Project (`pip install -e .`)
5. Freeze requirements (`pip freeze > requirements.txt`)
6. Install requirements for development (`pip install .[dev]`)
"""


# %%
### Builtin imports
import subprocess
import shlex
import os
import venv

### Third party imports, requires prepare_venv.py to be run
import toml
from icecream import ic  # * For Testing these functions

# from env_config import *


### For testing purposes, define location of pyproject.toml
if __name__ == "__main__":
    project_root = ".."
    pyproject_toml = os.path.join(project_root, "pyproject.toml")


# %%
def run_cmd(cmd: str, print_cmd=True, ret_p=False, **kwargs):
    """Run a command in the shell, e.g. `brew install python@3.11`"""

    ### Define default kwargs and override with user kwargs
    kws = dict(
        shell=False,  # !! Avoid shell, defaults to /bin/sh
    )
    kws.update(kwargs)

    ### Print command
    if print_cmd:
        print("\n$", cmd)

    ### Split command into list
    cmd: str = shlex.split(cmd)

    ### Run command
    #' .Popen doesn't output files as easily
    p = subprocess.run(cmd)

    if ret_p:
        return p


if __name__ == "__main__":
    c = "ping -c 4 google.com"
    # c = "echo $PATH"
    # c = "pip install pyperclip"
    run_cmd(c)

    # %%
    # ! pip uninstall pyperclip -y
    c = "pip install pyperclip"
    run_cmd(c)


# %%
def check_python(toml_f: str) -> None:
    print("We need Python Version: ", get_python_version(toml_f))
    run_cmd("python --version")
    run_cmd("which python")


def check_path():
    print("Checking PATH:")
    PATH = subprocess.check_output(["echo $PATH"], shell=True)
    PATH_list = PATH.decode("utf-8").split(":")
    for path in PATH_list:
        print(path)
    print(len(PATH_list))


# %%
def create_venv(project_root: str, env_name: str) -> None:
    """Create a virtual environment for the project"""
    if os.path.isdir(os.path.join(project_root, env_name)):
        print(f"'{env_name}' will be overwritten!")
    print(f">>> venv.create({env_name}, clear=True, with_pip=True)")
    venv.create(
        env_dir=os.path.join(project_root, env_name),
        clear=True,  #' Clear the environment directory before creating it
        with_pip=True,
    )


def make_python_path(project_root: str, env_name: str) -> str:
    """Get the path to the python interpreter of the virtual environment"""
    return os.path.join(project_root, env_name, "bin", "python")


# %%
def install(packages: list, python: str = None):
    """Install packages"""
    python = "python" if python is None else python

    packages_str = " ".join(packages)
    run_cmd(f"{python} -m pip install {packages_str}")


def install_deps(
    packages: list = None, requirements_txt: str = None, python: str = None
) -> None:
    """Install dependencies from requirements.txt, if provided, else
    from pyproject.toml
    """
    python = "python" if python is None else python

    if not requirements_txt is None:
        assert os.path.isfile(requirements_txt), f"{requirements_txt} not found"
        run_cmd(f"{python} -m pip install -r {requirements_txt}")

    else:
        install(packages, python=python)


# %%
def toml_to_dict(toml_f: str) -> dict:
    """Convert toml file to dict"""
    with open(toml_f, "r") as f:
        pypr: dict = toml.load(f)
    return pypr


if __name__ == "__main__":
    pypr = toml_to_dict(pyproject_toml)
    ic(pypr["project"])


# %%
def get_project_name(toml_f: str) -> str:
    """Get the project name from pyproject.toml"""
    pypr = toml_to_dict(toml_f=toml_f)
    return pypr["project"]["name"]


def get_python_version(toml_f: str) -> str:
    """Get the python version from pyproject.toml"""
    pypr = toml_to_dict(toml_f=toml_f)
    return pypr["project"]["requires-python"]


if __name__ == "__main__":
    r = get_python_version(pyproject_toml)
    ic(r)


# %%
def get_dependencies(toml_f: str, optional: str = None) -> list:
    """Get the dependencies from pyproject.toml"""
    pypr = toml_to_dict(toml_f=toml_f)
    if optional is None:
        return pypr["project"]["dependencies"]
    else:
        keys = list(pypr["project"]["optional-dependencies"].keys())
        if optional in keys:
            return pypr["project"]["optional-dependencies"][optional]
        else:
            print(
                f"""Optional dependency '{optional}' not found, use one of: {keys}"""
            )


if __name__ == "__main__":
    r = get_dependencies(pyproject_toml)
    ic(r)

    # %%
    r = get_dependencies(pyproject_toml, optional="dev")
    ic(r)
    # %%
    r = get_dependencies(pyproject_toml, optional="devvv")
    ic(r)


# %%
def print_header(s: str, line_char="=") -> None:
    """Print a header with a string in the middle Adds two = before the
    string and then adds = after the string until total length of string
    is 80 reached
    """
    print("\n")  # ' two empty lines
    print(line_char * 2, s, line_char * (80 - len(s) - 2))
    print()
    
def print_dict(d: dict) -> None:
    """Print a dictionary for terminal nice and pretty with indentation
    and adjusted spacing between keys and values"""
    for k, v in d.items():
        k = str(k) + ":"
        #' ljust between the spaces "venv (passed: venv)""
        v = str(v).split(" ")
        v = v[0].ljust(8) + " ".join(v[1:])
        print(f"    {k.ljust(30)} {v}")