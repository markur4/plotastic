#
# == Metadata ==========================================================
from importlib import metadata

# ? https://packaging.python.org/guides/single-sourcing-package-version/
# ? Do we need this?
__version__ = metadata.version(__name__)
__author__ = "markur4"


# == Flatten module access  ============================================
from .plotting.rc_utils import set_style, print_styles, set_palette
from .dataanalysis.dataanalysis import DataAnalysis
from .example_data.load_dataset import load_dataset
from .utils import utils


# == __all__ ===========================================================
__all__ = [
    DataAnalysis,
    set_style,
    print_styles,
    set_palette,
    load_dataset,
]
