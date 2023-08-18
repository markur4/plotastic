import pandas as pd

from plotastic.assumptions import Assumptions


class Omnibus(Assumptions):
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)
