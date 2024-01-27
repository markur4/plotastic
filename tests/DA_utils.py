"""A utility class that creates DataAnalysis objects for testing"""
# %%


import pandas as pd

import plotastic as plst
from plotastic.dataanalysis.dataanalysis import DataAnalysis
from plotastic.utils.subcache import SubCache

import DA_configs as dac

# %%
# == Class CreateDA ====================================================


class TestDA(DataAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        subject: str = None,
        levels: list[tuple[str]] = None,
        title: str = "untitled",
        verbose=False,
    ) -> DataAnalysis:
        kws = dict(
            data=data,
            dims=dims,
            subject=subject,
            levels=levels, #' Introduced by DataFrameTool
            title=title, #' Introduced by DataAnalysis
            verbose=verbose, #' Introduced by DataAnalysis
        )

        super().__init__(**kws)
        
    def perform_statistics_unpaired(self, parametric=True) -> "TestDA":
        """Perform unpaired statistics"""
        ### Assumptions
        self.check_normality()
        self.check_homoscedasticity()
        
        ### Omnibus
        if parametric:
            self.omnibus_anova()
        else:
            self.omnibus_kruskal()
        
        ### PostHoc
        self.test_pairwise(parametric=parametric)
        
        return self
    
    def perform_statistics_paired(self, parametric=True) -> "TestDA":
        """Perform unpaired statistics"""
        ### Assumptions
        self.check_normality()
        self.check_homoscedasticity()
        self.check_sphericity()
        
        ### Omnibus
        if parametric:
            self.omnibus_anova()
        else:
            self.omnibus_kruskal()
        
        ### PostHoc
        self.test_pairwise(parametric=parametric)
        
        return self



if __name__ == "__main__":
    pass
    # %%
    dims = dac.dims_withempty_tips[0]
    data = dac.DF_tips
    DA = TestDA(data=data, dims=dims)
    