import pandas as pd

from plotastic.dimsandlevels import DimsAndLevels
from plotastic.statresults import StatResults


class Assumptions(DimsAndLevels):
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)

        self.results = StatResults()
