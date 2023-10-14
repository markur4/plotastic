from importlib import metadata
import glob
from pathlib import Path


# from os.path import dirname, basename, isfile, join
# import os

# from markurutils.enviroments import check_dependencies

# == Metadata ==========================================================================
__name__ = "plotastic"
__version__ = metadata.version(__name__)
__author__ = "markur4"

# == Automated imports =================================================================
# https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
### Get subdirectories that store all .py Files. Exclude hidden folders
# cwd = Path(__file__).parent
# directories = [
#     f for f in cwd.iterdir() if f.is_dir() and not f.name.startswith(("_", "."))
# ]
# ### Get all .py files in subdirectories
# module_paths = []
# for dir in directories:
#     module_paths += glob.glob(str(dir / "*.py"))

# ### Put all modules in __all__. Take care to include parent folders of modules
# __all__ = []
# for f in module_paths:
#     if "__init__" in f:
#         continue
#     else:
#         module = f"{Path(f).parent.stem}.{Path(f).stem}"
#         __all__.append(module)

# ### cleanup
# del cwd, module_paths

#
# == Check Dependencies ================================================================
# missing_hard = check_dependencies(deps=requirements, hard=True)

#
# == Flatten module access and import everything =======================================
# * take care to include parent folders of modules

# ! This works, but vscode won't recognize and highlight functions and classes
# ! Still keep it, to make sure everything is imported
# for module in __all__:
#     exec(f"from .{module} import *")

### Vscode requires explicit imports for syntax highlighting

from .dimensions.dims import Dims
from .dimensions.dimsandlevels import DimsAndLevels
from .dimensions.dataframetool import DataFrameTool

from .plotting.rc import set_style, set_palette
from .plotting.plottool import PlotTool
from .plotting.plotedits import PlotEdits
from .plotting.multiplot import MultiPlot

from .stat.statresults import StatResults
from .stat.stattest import StatTest
from .stat.assumptions import Assumptions
from .stat.bivariate import Bivariate
from .stat.posthoc import PostHoc
from .stat.omnibus import Omnibus

from .dataanalysis.annotator import Annotator
from .dataanalysis.dataanalysis import DataAnalysis

### Use __all__ to let type checkers know what's available
__all__ = [
    DataAnalysis,
    set_style,
    set_palette,
    DataFrameTool,
    Dims,
    DimsAndLevels,
    PlotTool,
    PlotEdits,
    MultiPlot,
    StatResults,
    StatTest,
    Assumptions,
    Bivariate,
    PostHoc,
    Omnibus,
    Annotator,
]

# == Cleanup Namespace ....................................................... """
# del requirements, missing_hard, #tobe_linked, source, path
