# %%
import markurutils as ut
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from analysis import Analysis


class PlotHelper(Analysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict,
        verbose=True,
        plot=None,
    ):
        super().__init__(data=data, dims=dims, verbose=verbose)

        self.plot = plot

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
### Make PlotHelper Object
stat = PlotHelper(
    data=DF,
    dims=DIMS,
    plot=g,  # * Only works with Facetgrid
)
print(stat)
# %%

stat.show_plot()

# %%
