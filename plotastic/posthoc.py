import pandas as pd
import pingouin as pg

# from plotastic.analysis import Analysis
from plotastic.assumptions import Assumptions


class PostHoc(Assumptions):
    # ... __INIT__ .......................................................#
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)

    # ... PAIRED T TEST ..................................................#
    
    def _base_test_multiple_paired_t(self, f1, f2=None, **kwargs) -> pd.DataFrame:
        
        ### Manage between or within
        pass
        
    
    def test_multiple_paired_t(self) -> pd.DataFrame:
        pass
