#!/usr/bin/env python
# coding: utf-8

# %%
import pandas as pd
import markurutils as ut
import plotastic as plst

print(plst.__all__)
# from plotastic import *
# Dims
from plotastic.dims import Dims
# from plotastic.dimsandlevels import DimsAndLevels


# %%
### Generate test data
df, dims = ut.load_dataset("fmri")
ut.pp(df.head())  # pp() just prints out dataframes


# %%

### Dims
# * Dims Objects store `{x, y, hue, col, row}`. It also has the property `by` that is by default a list of `[row, col]`. This is needed to group the dataframe just as facetgrid does"

print("\n# Dims Objects store x, y, hue, col, row")
dims = Dims(**dict(y="signal", x="timepoint", hue="event", col="subject"))
dims

# In[7]:

### Switch
# * DIMS Objects can switch x, y, hue, col, row by changing the dims object either inplace or by generating a copy"

dims = Dims(**dict(y="signal", x="timepoint", hue="event", col="subject"))

print(
    "#\t 'dims.by' before switching:",
    dims.col,
)
print("#\t 'dims.by' in chain:        ", dims.switch(col="event", inplace=False).col)
print("#\t 'dims.by' after switching: ", dims.col, "(inplace=False)")
print("#\t 'dims.by' in chain:        ", dims.switch(col="event", inplace=True).col)
print("#\t 'dims.by' after switching: ", dims.col, "(inplace=True)")


# - Switching method can be called passing in two *args without knowing the column names. One of them may be one of `[y,x,hue,row,col]`, which is helpful if you don't remember the exact column names "

# In[5]:

# * Switching method can be called passing in two *args without knowing the column names One of them may be one of [y,x,hue,row,col]"

print("#\t 'dims.by' during witching: ", dims.switch(col="event", inplace=False).col)
print("#\t 'dims.by' after switching: ", dims.col, "(inplace=False)")


# # Analysis
#
# ##### Analysis class is composed by:
# - **Dims object**
# - **Pandas Dataframe** for Raw Data
# - **Filer Class** Is automatically set. Determines output path (and input path if pyrectories is used)
# - **Project Title** `(str)` to use as figure suptitles and filenames for outputs
# - **properties:**
#     - factors_all: Stores all factors `{x, y, hue, col, row}`
# - **methods**
#     - getfactors(): Retrieves the column name when one of `{x, y, hue, col, row}` is passed
#     - switch(), set() is callable on Analysis object (same funitonality as dims)

# In[ ]:


df, _ = ut.load_dataset(
    "fmri"
)  # * _ stores a dictionary with dims. For clarity, we initialize analysis explicitly
ana = ut.Analysis(
    dims=dict(y="signal", x="timepoint", hue="event", col="subject"),
    title="APROJECT",
    data=df,
)


# In[8]:


print(
    "\n### ANALYSIS Objects can also switch x, y, hue, col, row by changing the dims object either inplace or by generating a copy"
)

df, _ = ut.load_dataset(
    "fmri"
)  # _ stores a dictionary with dims. For clarity, we initialize analysis explicitly
ana = ut.Analysis(
    dims=dict(y="signal", x="timepoint", hue="event", col="subject"),
    title="APROJECT",
    data=df,
)

print(
    "#\t 'ana.dims.by' before switching:",
    ana.dims.by,
)
print(
    "#\t 'ana.dims.by' in chain:        ",
    ana.switch(col="event", inplace=False).dims.by,
)
print("#\t 'ana.dims.by' after switching: ", ana.dims.by, "(inplace=False)")
print(
    "#\t 'ana.dims.by'  in chain:       ", ana.switch(col="event", inplace=True).dims.by
)
print("#\t 'ana.dims.by' after switching: ", ana.dims.by, "(inplace=True)")


# In[6]:
