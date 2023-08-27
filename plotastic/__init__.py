from importlib import metadata
import glob
from pathlib import Path

from pytools import P

# from os.path import dirname, basename, isfile, join
# import os

# from markurutils.enviroments import check_dependencies

# ... Metadata .....................................................
__name__ = "plotastic"
__version__ = metadata.version(__name__)
__author__ = "markur4"

# ... Specify what to import when using import * .....................................................
# *  https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
### List all .py files in this directory and subdirectories
cwd = Path(__file__).parent
directories = [
    f for f in cwd.iterdir() if f.is_dir() and not f.name.startswith(("_", "."))
]
module_paths = []
for dir in directories:
    module_paths += glob.glob(str(dir / "*.py"))

### Put all modules in __all__ except __init__.py. Take care to include parent folders of modules
# __all__ = [Path(f).stem for f in module_paths if "__init__" not in f]
__all__ = []
for f in module_paths:
    if "__init__" in f:
        continue
    else:
        module = f"{Path(f).parent.stem}.{Path(f).stem}"
        __all__.append(module)

### cleanup
del cwd, module_paths

# ... Check Dependencies .....................................................
# missing_hard = check_dependencies(deps=requirements, hard=True)


# ... Flatten module access and import everything .....................................................
# * take care to include parent folders of modules


for module in __all__:
    exec(f"from .{module} import *")

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
