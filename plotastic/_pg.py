
# %%
import markurutils as ut
import seaborn as sns
import seaborn.objects as so
import pandas as pd

from IPython.display import display

from analysis import Analysis

#%%

class Statter(Analysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        title: str,
        plot: so.Plot | sns.axisgrid.FacetGrid = None,
    ):
        super().__init__(data, dims, title)
        self.plot = plot
        ### Statistics
        # self.test = Test()

    def show_plot(self):
        display(self.plot)



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
g = sns.catplot(kind="swarm", data=DF, **DIMS)
print(g)

# plot = (  #! not working with seaborn objects
#     so.Plot(data=DF, x=DIMS["x"], y=DIMS["y"], color=dims["hue"])
#     .add(so.Line)
#     .facet(row=dims["row"], col=dims["col"])
# )

# %%
### Make Statter
stat = Statter(
    data=DF,
    dims=DIMS,
    title="Tips Analysis",
    plot=g,  # * Only works with Facetgrid
)
print(stat)
# %%

stat.show_plot()

# %%

grouped = DF.groupby([#DIMS["col"], DIMS["row"], 
                      DIMS["hue"], DIMS["x"]], dropna=False)

for name, df in grouped:
    print(name) # *skips empty frames!


newDF = stat.data_ensure_allgroups()


# %%
grouped = newDF.groupby([
    # DIMS["col"], DIMS["row"], 
    DIMS["hue"], DIMS["x"]], dropna=False)

# grouped.get_group(('Sat', 'Lunch', '1-2', 'Yes')) #* WORKING NOW !!!
# grouped.get_group(('Sat', 'Dinner', '1-2', 'Yes')) #* works, not empty

# %%

stat.plot_data()
stat.describe_data()


# %%
##%! timeit
#% 
stat.get_empties()
                 
# %%
