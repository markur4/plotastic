# %%

# %%
import markurutils as ut
import seaborn as sns

# import seaborn.objects as so
import pandas as pd

from IPython.display import display

from analysis import Analysis
from plothelper import PlotHelper


# %%
### Load Data
DF, dims = ut.load_dataset("tips")
# DIMS = dims
DIMS = dict(y="tip", x="smoker", hue="size-cut", col="day", row="time")
ut.pp(DF.head(5))
print(DIMS)


# %%
### Make Analysis Object
A = Analysis(data=DF, dims=DIMS, title="Tips Analysis", verbose=True)
# %%
### Show those groups that are empty
A.data_get_empty_groupkeys()

# %%
### Show content of Analysis object
print(A)

# %%
### Make a basic plot
A.plot_quick()

# %%

### Describe the data
A.data_describe()

# %%
