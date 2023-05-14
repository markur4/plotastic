from importlib import metadata
# from markurutils.enviroments import check_dependencies

'''# Metadata .....................................................'''
__name__            = "plotastic"
__version__         = metadata.version(__name__)
__author__          = 'markur4'


'''# Check Dependencies .....................................................'''
# missing_hard = check_dependencies(deps=requirements, hard=True)


'''# Flatten module access .....................................................'''
from .analysis import *
from .dims import *

from .plothelper import *
from .plotsnippets import *

from .stattester import *
from .assumptions import *
from .posthoc import *
from .omnibus import *

from .dataanalysis import *

from ._pg import *

'''# Cleanup Namespace ....................................................... '''
#del requirements, missing_hard, #tobe_linked, source, path
