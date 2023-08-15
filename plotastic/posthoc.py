import pandas as pd

# from plotastic.analysis import Analysis
from plotastic.assumptions import Assumptions


class PostHoc(Assumptions):
    # ... __INIT__ .......................................................#
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)

    # ... PAIRED T TEST ..................................................#
    def test_multiple_paired_t(self):
        pass
