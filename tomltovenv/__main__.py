#
# %%

import subprocess
import os
import argparse

# !! This must be the only place where rc is imported, otherwise user
# !! arguments will be overwritten if called again later
from rc import (
    PROJECT_ROOT,
    VENV_NAME,
    REDO_REQUIREMENTS_TXT,
    EDITABLE,
    INSTALL_DEVTOOLS,
)

# %%

# == INIT Parsers ==============================================================
parser = argparse.ArgumentParser(
    description="Reads your ``pyproject.toml`` inside project root and sets up a virtual environment including dependencies, devtools, editable install, etc.",
    add_help=True,
)

# ==============================================================================
# == Add Arguments to Parser ===================================================
# fmt: off

### Reused args
kws = dict(
    default=None,  
    required=False,
    action="store_true",
)

### Optional
parser.add_argument(
    "-p", "--project-root", 
    help="Path to project root. Default: %(default)s",
    type=str, #' Cast type when passing
    default=".",  #' Ok, since we assert for presence of pyproject.toml
)
parser.add_argument(
    "-v", "--venv-name",
    help="Name of virtual environment. Default: %(default)s",
    type=str,
    default="venv",
)

### Flags
parser.add_argument(
    "-r", "--redo-requirements",
    help="!! Use Re-solve environment and (over)write requirements.txt. Default: %(default)s",
    **kws,
)
parser.add_argument(
    "-t", "--test-dir",
    help="Specify directory within project root to find tests. If not specified, no tests will be run. If only -t is passed without directory, will use 'tests' as values. Requires --install-devtools. Default: %(default)s",
    type=str,
    nargs="?",  #' Make optional
    const=True,  #' If only -t is passed without directory, set to True
    default=False, # !! overriden later to "tests"
)

### --user-mode switches off Features
#' User mode: Mimic a setup of a user after pip install <package>
#' User mode: Switches off devtools, editable
parser.add_argument(
    "-u", "--user-mode",
    help="Executes standard install of the project (pip install <project>), not editable and without devtools. Default: %(default)s",
    **kws,
)
#' Switch back on despite --user-mode
parser.add_argument(
    "-e", "--editable",
    help="By default, %(prog)s installs project in editable mode, but when switched off by --user-mode, use --editable to switch it back on",
    **kws,
)
parser.add_argument(
    "-d", "--install-devtools",
    help="By default, %(prog)s installs developer tools, but when switched off by --user-mode, use --editable to switch it back on",
    **kws,
)
# fmt: on


# ==============================================================================
# == Redirect user input into settings =========================================
#' Values from rc.py are capitalized: PROJECT_ROOT, VENV_NAME, etc.

### Parse Args
args = parser.parse_args()

### Paths
# if args.project_root:  #' -p, string, default to "."
PROJECT_ROOT = args.project_root

# if args.venv_name:  #' -v, string, default to "venv"
VENV_NAME = args.venv_name

### Flags
if not args.redo_requirements is None:  #' -r
    REDO_REQUIREMENTS_TXT = args.redo_requirements

if args.test_dir is True:
    test_dir = "tests" #' -t
else:
    test_dir = args.test_dir


### --user-mode switches off Features
# !! -e and -d default to False, but we don't expect the user to use
# !! them without --user-mode
EDITABLE = True
INSTALL_DEVTOOLS = True
if args.user_mode:
    EDITABLE = False  #' -e default to False
    INSTALL_DEVTOOLS = False  #' -d default to False

### Fine-tune User Mode.
#' Allow to switch specific options back on
if not args.editable is None:  #' -e
    EDITABLE = args.editable

if not args.install_devtools is None:  #' -d
    INSTALL_DEVTOOLS = args.install_devtools


# ==============================================================================
def args_to_dict()-> dict:
    return {
        "Project root": f"{PROJECT_ROOT} (-p '{args.project_root}')",
        "Venv name": f"{VENV_NAME} (-v {args.venv_name})",
        "Redo requirements.txt": f"{REDO_REQUIREMENTS_TXT} (-r {args.redo_requirements})",
        f"Editable install of project": f"{EDITABLE} (-e {args.editable})",
        "Install devtools": f"{INSTALL_DEVTOOLS} (-d {args.install_devtools})",
        "Test directory": f"{test_dir} (-t '{args.test_dir}')",
    }

# :: Uncomment to test before assertion
# from pprint import pprint
# pprint(args_to_dict())

# ==============================================================================
# == Assert Correct Inputs =====================================================

### Construct expected locations of Metadata
pyproject_toml = os.path.join(PROJECT_ROOT, "pyproject.toml")
requirements_txt = os.path.join(PROJECT_ROOT, "requirements.txt")

### Find pyproject.toml
project_root_full = os.path.abspath(args.project_root)
assert os.path.exists(
    pyproject_toml
), f"Didn't find pyproject.toml in project root ({project_root_full})"

### Tests without devtools
assert not (
    test_dir != "Don't test" and not INSTALL_DEVTOOLS
), "Cannot run tests without installing devtools, use -d / --install-devtools"

### Tests without specifying directory
if test_dir:
    test_path = os.path.join(PROJECT_ROOT, test_dir)
    assert os.path.exists(
        test_path
    ), f"'{test_dir}' not found in project root ({project_root_full}), but -t / --run-tests was specified"


def main():
    # ==========================================================================
    # == Run
    # ==========================================================================
    #
    # from pprint import pprint

    ### Import utils
    #' Install Dependencies for tomltovenv, since utils_ttv needs that
    subprocess.run(
        ["python", os.path.join(PROJECT_ROOT, "tomltovenv", "prepare.py")]
    )
    import utils_ttv as ut

    # %%
    ### Print Arguments:
    ut.print_header("SETTINGS:", line_char="#")
    ut.print_dict(args_to_dict())

    # hä
    # %%
    ### Check which python executable is being used
    ut.print_header("Checking Python Version")
    ut.check_python(pyproject_toml)  # Check Python version

    #' Install global Python
    cmd = "brew install python@3.11"
    # eut.run_cmd(cmd) # :: uncomment

    # %%
    ut.print_header(f"Creating Virtual Environment: '{VENV_NAME}' ")
    ut.create_venv(PROJECT_ROOT, VENV_NAME)

    # %%
    ### Construct path to python executable in venv
    python = ut.make_python_path(PROJECT_ROOT, VENV_NAME)
    python

    # %%
    def mp(module: str) -> str:
        """Returns path to the tomltovenv module"""
        return os.path.join(PROJECT_ROOT, "tomltovenv", module)

    # %%
    ### Install Dependencies for tomltovenv
    ut.run_cmd(f"{python} {mp('prepare.py')}", print_cmd=False)

    # %%
    ### Install Dependencies for PROJECT
    ut.print_header(
        f"Installing Dependencies for {ut.get_project_name(pyproject_toml)}"
    )
    argv = [PROJECT_ROOT, VENV_NAME, REDO_REQUIREMENTS_TXT, INSTALL_DEVTOOLS]
    argv = " ".join([str(arg) for arg in argv])
    ut.run_cmd(f"{python} {mp('dependencies.py')} {argv}")

    # %%
    ### Install this PROJECT
    if EDITABLE:
        ut.print_header(
            f"Installing Editable {ut.get_project_name(pyproject_toml)}"
        )
        ut.run_cmd(f"{python} -m pip install -e {PROJECT_ROOT}")
    else:
        ut.print_header(
            f"Installing {ut.get_project_name(pyproject_toml)} (Not Editable)"
        )
        ut.run_cmd(f"{python} -m pip install {PROJECT_ROOT}")

    # %%
    ### Run tests to check environment
    if INSTALL_DEVTOOLS and test_dir:
        ut.print_header(f"Running Tests")
        ut.run_cmd(f"{python} -m pytest {test_dir}")


if __name__ == "__main__":
    main()
