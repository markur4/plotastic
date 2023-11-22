"""Installs dependencies for the project and freezes them to a
requirements.txt file."""


# %%
import os
import subprocess

import utils_ttv as ut

# from rc import *


def install(
    project_root: str,
    venv_name: str,
    redo_requirements_txt: bool,
    install_devtools: bool,
) -> None:
    # %%
    ### Construct expected locations of Metadata
    pyproject_toml = os.path.join(project_root, "pyproject.toml")
    requirements_txt = os.path.join(project_root, "requirements.txt")

    # %%
    ### Check Dependencies
    deps = ut.get_dependencies(pyproject_toml)
    deps_dev = ut.get_dependencies(pyproject_toml, optional="dev")
    print()
    print("Dependencies: ", deps)
    print("Developer tools: ", deps_dev)

    # %%
    ### Check requirements.txt
    # - requirements.txt represents solved environment
    # - No developer tools
    print()
    print(requirements_txt, "exists: \t", os.path.exists(requirements_txt))
    do_req = redo_requirements_txt or not os.path.exists(requirements_txt)
    print("Make requirements.txt:\t\t", do_req)
    overwrite = redo_requirements_txt and os.path.exists(requirements_txt)
    print("Overwriting requirements.txt:\t", overwrite)

    # %%
    ### Check python venv
    python = ut.make_python_path(project_root, venv_name)
    # print("Python executable: ", python)

    # %%
    ### Install Dependencies
    #' Not providing requirements.txt will pip re-solve the environment
    if do_req:
        print("Installing from: \tdependency list from pyproject.toml")
        ut.install_deps(packages=deps, python=python)
    else:
        print("Installing from: \trequirements.txt")
        ut.install_deps(requirements_txt=requirements_txt, python=python)

    # %%
    ### Freeze Environment
    if do_req:
        ut.print_header("Freezing environment")
        cmd = f"{python} -m pip freeze > {requirements_txt}"
        print("\n$ " + cmd)
        os.system(cmd)
        # subprocess.run([python, "-m", "pip", "freeze", ">", REQUIREMENTS_TXT])

    # %%
    ### Install Developer Tools
    if install_devtools:
        ut.print_header("Installing developer tools")
        ut.install_deps(packages=deps_dev, python=python)


if __name__ == "__main__":
    
    ### This gets called by main()
    import sys
    install(
        project_root=sys.argv[1],
        venv_name=sys.argv[2],
        redo_requirements_txt=True if sys.argv[3] == "True" else False,
        install_devtools=True if sys.argv[4] == "True" else False,
    )
