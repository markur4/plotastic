

# %% Import Data
import plotastic as plst
DF, dims= plst.load_dataset("fmri")

dims

# %% plot
import seaborn as sns
sns.catplot(data=DF, **dims, kind="box")


# %% Check functionality with pingouin
import pingouin as pg
pg.normality(DF, dv=dims["y"], group=dims["x"])
pg.homoscedasticity(DF, dv=dims["y"], group=dims["x"])

# %% create Assumptions object
from plotastic import Assumptions
DA = Assumptions(data=DF, dims=dims)

# %% Check normality
DA.check_normality()

# %% Check homoscedasticity
DA.check_homoscedasticity()

# %%
