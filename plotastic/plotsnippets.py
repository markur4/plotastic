
from plotastic.analysis import Analysis


class PlotSnippets(Analysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        title: str,
        verbose = True
    ):
        super().__init__(data=data, dims=dims, title=title, verbose=verbose)