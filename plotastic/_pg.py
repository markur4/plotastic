
# %%
import markurutils as ut
import seaborn as sns
# import seaborn.objects as so
import pandas as pd

from IPython.display import display

from analysis import Analysis


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
    # plot=g,  # * Only works with Facetgrid
)
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


newDF = A.data_ensure_allgroups()



# %%
##%! timeit
#% 

                 
# %%



class DataAnalysis(Analysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        title: str,
    ):
        ### Inherit
        super().__init__(data, dims, title)
        ### Check Data
        self.get_empties(verbose=True)
        
        
        self.plot = plot
        ### statistics
        # self.test = Test()

    def show_plot(self):
        display(self.plot)

