from importlib import metadata

# == Add Icecream to builtins# =================================================
try:
    from icecream import ic, install

    install()
except ImportError:  # # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


# == Metadata ==================================================================
__version__ = metadata.version(__name__)
__author__ = "markur4"


#
# == Flatten module access and import everything ===============================

# from .dimensions.dims import Dims
# from .dimensions.dimsandlevels import DimsAndLevels
# from .dimensions.dataframetool import DataFrameTool

from .plotting.rc import set_style, set_palette

# from .plotting.plottool import PlotTool
# from .plotting.plotedits import PlotEdits
# from .plotting.multiplot import MultiPlot

# from .stat.statresults import StatResults
# from .stat.stattest import StatTest
# from .stat.assumptions import Assumptions
# from .stat.bivariate import Bivariate
# from .stat.posthoc import PostHoc
# from .stat.omnibus import Omnibus

from .dataanalysis.dataanalysis import DataAnalysis

# from .dataanalysis.annotator import Annotator
# from .dataanalysis.filer import Filer

from .example_data.load_dataset import load_dataset

# == __all__ ===================================================================
__all__ = [
    DataAnalysis,
    set_style,
    set_palette,
    load_dataset,
    # DataFrameTool,
    # Dims,
    # DimsAndLevels,
    # PlotTool,
    # PlotEdits,
    # MultiPlot,
    # StatResults,
    # StatTest,
    # Assumptions,
    # Bivariate,
    # PostHoc,
    # Omnibus,
    # Annotator,
    # Filer,
]
