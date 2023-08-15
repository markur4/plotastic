import pandas as pd
from plotastic.analysis import Analysis
from plotastic.assumptions import Assumptions


class PostHoc(Assumptions):
    def __init__(self, data: pd.DataFrame, dims: dict, verbose=False):
        super().__init__(data=data, dims=dims, verbose=verbose)
