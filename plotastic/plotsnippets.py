import pandas as pd

from analysis import Analysis


class PlotSnippets(Analysis):
    def __init__(self, data: pd.DataFrame, dims: str, verbose=False):
        super().__init__(data=data, dims=dims, verbose=verbose)
