import pandas as pd

from plotastic.analysis import Analysis
from plotastic.plottool import PlotTool


class MultiPlot(PlotTool):
    def __init__(self, data: pd.DataFrame, dims: str, verbose=False):
        super().__init__(data=data, dims=dims, verbose=verbose)
