"""Adds Subject funcitonality to DataAnalysis."""
# %%
#== Imports ============================================================

# from plotastic
from plotastic.utils import utils as ut
from plotastic.dimensions.dimsandlevels import DimsAndLevels

# %%
#== Class Subject ======================================================

class Subject (DimsAndLevels):
    
    def __init__(self, subject = None, **kws) -> None:
        super().__init__(**kws)
        self.subject = subject
        if not subject is None:
            assert (
                subject in self.data.columns
            ), f"#! Subject '{subject}' not in columns, expected one of {self.data.columns.to_list()}"
        
    @property
    def subjectlist(self):
        if self.subject is None:
            raise TypeError("No subject column specified")
        return tuple(self.data[self.subject].unique())

