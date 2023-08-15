#
#%% Imports
import markurutils as ut
import plotastic as plst

# %% Get Data and make DataAnalysis object

DF, dims = ut.load_dataset("fmri")
DA = plst.DataAnalysis(data=DF, dims=dims, subject="subject")


# %% plot Data
DA.plot()




# %%
