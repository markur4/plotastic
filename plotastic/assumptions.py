import pandas as pd
from analysis import Analysis
from stattester import StatTester


class Assumptions(StatTester):
    def __init__(self, data: pd.DataFrame, dims: dict, title: str, verbose=False):
        super().__init__(data=data, dims=dims, title=title, verbose=verbose)
