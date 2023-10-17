from importlib import metadata
# from markurutils.enviroments import check_dependencies

'''# Metadata .....................................................'''
__name__            = "markurutils"
__version__         = metadata.version(__name__)
__author__          = 'markur4'


'''# Check Dependencies .....................................................'''
# missing_hard = check_dependencies(deps=requirements, hard=True)


'''# Flatten module access .....................................................'''
# from markurutils import UTILS
# from markurutils import filer
# from markurutils import export
# from markurutils import analysis
# from markurutils import setup_envs

from .UTILS import *
from .builtin_types import *
from .modules import *
# from .analysis import Analysis
from .filer import *
from .exports import  *
from .enviroments import *


'''# Cleanup Namespace ....................................................... '''
#del requirements, missing_hard, #tobe_linked, source, path
