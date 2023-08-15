from importlib import metadata
import glob
from pathlib import Path
# from os.path import dirname, basename, isfile, join
# import os

# from markurutils.enviroments import check_dependencies

# ... Metadata .....................................................
__name__ = "plotastic"
__version__ = metadata.version(__name__)
__author__ = "markur4"

# ... Specify what to import when using import * .....................................................
# *  https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
### List all .py files in this directory
cwd = Path(__file__).parent
modules = glob.glob(str(cwd / "*.py"))
### Put all modules in __all__ except __init__.py
__all__ = [
    Path(f).stem for f in modules if Path(f).exists() and not f.endswith("__init__.py")
]

### cleanup
del cwd, modules

# ... Check Dependencies .....................................................
# missing_hard = check_dependencies(deps=requirements, hard=True)


# ... Flatten module access .....................................................
# from .analysis import *
# from .dims import *

# from .assumptions import *
# from .posthoc import *
# from .omnibus import *
# # from .old.statresult import *

# from .plottool import *
# from .multiplot import *

# from .dataanalysis import *

# from ._pg import *

# ... Cleanup Namespace ....................................................... """
# del requirements, missing_hard, #tobe_linked, source, path
