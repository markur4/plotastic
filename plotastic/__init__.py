from importlib import metadata

# from markurutils.enviroments import check_dependencies

"""# Metadata ....................................................."""
__name__ = "plotastic"
__version__ = metadata.version(__name__)
__author__ = "markur4"


"""# Check Dependencies ....................................................."""
# missing_hard = check_dependencies(deps=requirements, hard=True)


"""# Flatten module access ....................................................."""
from .analysis import *
from .dims import *

from .assumptions import *
from .posthoc import *
from .omnibus import *
# from .old.statresult import *

from .plottool import *
from .multiplot import *

from .dataanalysis import *

from ._pg import *

"""# Cleanup Namespace ....................................................... """
# del requirements, missing_hard, #tobe_linked, source, path
