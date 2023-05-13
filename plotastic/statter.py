# %%
import markurutils as ut
import seaborn as sns
import seaborn.objects as so
import pandas as pd

from IPython.display import display

from analysis import Analysis
from test import Test


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
        self.test = Test()

    def show_plot(self):
        display(self.plot)


### Testing #.......................................................................................................

# %%
### Load Data
DF, dims = ut.load_dataset("tips")
DIMS = dict(y="tip", x="sex", hue="size-cut", col="smoker", row="time")
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
