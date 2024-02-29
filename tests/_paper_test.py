### IMPORT PLOTASTIC
import plotastic as plst

# IMPORT EXAMPLE DATA
DF, _dims = plst.load_dataset("fmri", verbose=False)
# EXPLICITLY DEFINE DIMENSIONS TO FACET BY
dims = dict(
    y="signal",  # y-axis, dependent variable
    x="timepoint",  # x-axis, independent variable (within-subject factor)
    hue="event",  # color, independent variable (within-subject factor)
    col="region",  # axes, grouping variable
)
# INITIALIZE DATAANALYSIS OBJECT
DA = plst.DataAnalysis(
    data=DF,  # Dataframe, long format
    dims=dims,  # Dictionary with y, x, hue, col, row
    subject="subject",  # Datapoints are paired by subject (optional)
    verbose=False,  # Print out info about the Data (optional)
)
# STATISTICAL TESTS
DA.check_normality()  # Check Normality
DA.check_sphericity()  # Check Sphericity
DA.omnibus_rm_anova()  # Perform RM-ANOVA
DA.test_pairwise()  # Perform Posthoc Analysis
# PLOTTING
(
    DA.plot_box_strip().annotate_pairwise(  # Pre-built plotting function initializes plot  # Annotate results from DA.test_pairwise()
        include="__HUE"  # Use only significant pairs across each hue
    )
)


### BACK-CHECK
import seaborn as sns
sns.catplot(data=DF, **_dims)