import pandas as pd

from plotastic.dataframetool import DataFrameTool
from plotastic.statresults import StatResults


class Assumptions(DataFrameTool):
    def __init__(self, **dims_and_levels_kws):
        super().__init__(**dims_and_levels_kws)

        self.results = StatResults()
