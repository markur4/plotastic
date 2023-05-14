import pandas as pd
from analysis import Analysis
from stattester import StatTester


class PostHoc(StatTester):
    def __init__(self, data: pd.DataFrame, dims: dict, verbose=False):
        super().__init__(data=data, dims=dims, verbose=verbose)
