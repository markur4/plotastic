from importlib import metadata
import glob
from pathlib import Path


# from os.path import dirname, basename, isfile, join
# import os

# from markurutils.enviroments import check_dependencies

# ... Metadata ........................................................................................
__name__ = "plotastic"
__version__ = metadata.version(__name__)
__author__ = "markur4"

# ... Automated imports ......................................................
# https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
### Get subdirectories that store all .py Files. Exclude hidden folders
cwd = Path(__file__).parent
directories = [
    f for f in cwd.iterdir() if f.is_dir() and not f.name.startswith(("_", "."))
]
### Get all .py files in subdirectories
module_paths = []
for dir in directories:
    module_paths += glob.glob(str(dir / "*.py"))

### Put all modules in __all__. Take care to include parent folders of modules
__all__ = []
for f in module_paths:
    if "__init__" in f:
        continue
    else:
        module = f"{Path(f).parent.stem}.{Path(f).stem}"
        __all__.append(module)

### cleanup
del cwd, module_paths

#
# ... Check Dependencies ..............................................................................
# missing_hard = check_dependencies(deps=requirements, hard=True)

#
# ... Flatten module access and import everything .....................................................
# * take care to include parent folders of modules


for module in __all__:
    exec(f"from .{module} import *")

# * Explicitly import DataAnalysis class so that syntax highlighting works
from .dataanalysis.dataanalysis import DataAnalysis

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
