import pandas as pd

from plotastic.dataframetool import DataFrameTool
from plotastic.statresults import StatResults


class Assumptions(DataFrameTool):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

        self.results = StatResults()
