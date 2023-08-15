import pandas as pd

from plotastic.analysis import Analysis
from plotastic.statresults import StatResults


class Assumptions(Analysis):
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)

        self.results = StatResults()
