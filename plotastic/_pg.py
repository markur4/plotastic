

# %%
import markurutils as ut
import seaborn as sns
# import seaborn.objects as so
import pandas as pd

from IPython.display import display

from analysis import Analysis
from plotastic.plothelper import PlotHelper


# %%
### Load Data
DF, dims = ut.load_dataset("tips")
# DIMS = dims
DIMS = dict(y="tip", x="smoker", hue="size-cut", 
            col="day", row="time"
            )
ut.pp(DF.head(5))
print(DIMS)


# %%
### Make Analysis Object
A = Analysis(
    data=DF,
    dims=DIMS,
    title="Tips Analysis", 
    verbose=True
)
# %%
### Show those groups that are empty
A.get_empty_groupkeys()

#%%
### Show content of Analysis object
print(A)

# %%
### Make a basic plot
A.plot_data()

#%%

### Describe the data
A.describe_data()

# %%

  
# %%

from typing import Dict
from copy import copy

from plothelper import PlotHelper
from plotsnippets import PlotSnippets
from stattester import Assumptions, Omnibus, PostHoc


class DataAnalysis(Analysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        title: str,
        verbose = True
    ):
        ### Inherit
        init_kws = dict(data=data, dims=dims, title=title)
        super().__init__(self.init_kws)
        
        ### Tools
        self.plothelper = PlotHelper(**init_kws)
        self.plotsnippets = PlotSnippets(**init_kws)
        self.assumptions = Assumptions(**init_kws)
        self.omnibus = Omnibus(**init_kws)
        self.posthoc = PostHoc(**init_kws)
        
        
        if verbose:
            self.warn_about_empties_and_NaNs()
        
        
        self.plot = plot
        ### statistics
        # self.test = Test()

    # ... KEEP PARAMETERS SYNCED
    
    
    def set_dims(self, dims):
        self.dims = dims
        self.plothelper.dims = dims
        self.plotsnippets.dims = dims
        self.assumptions.dims = dims
        self.omnibus.dims = dims
        self.posthoc.dims = dims
    
    
    def switch(  
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> "Analysis":
        da = self if inplace else copy(self)

        # * NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply
        da.dims = da.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)

        return da

        
    
    @property
    def plothelper(self):
        return PlotHelper(self.init_kws)

    # ... PLOTTING
    def show_plot(self):
        display(self.plot)


# %%


