import pandas as pd

from plotastic.assumptions import Assumptions


class Omnibus(Assumptions):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)
