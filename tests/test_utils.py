"""Utilities for testing plotastic. Contains lists of arguments"""


# %% Arguments

### Args that make empty groups
DIMS_TIPS_EMPTYGROUP = [ 
    dict(y="tip", x="day", hue="sex", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="sex", hue="day", col="time"),
    dict(y="tip", x="sex", hue="day", row="time"),
    dict(y="tip", x="sex", hue="day", row="size-cut"),
    dict(y="tip", x="sex", hue="day"),
    dict(y="tip", x="sex"),
    dict(y="tip", x="size-cut"),
]

### Args making sure they don't make empty groups
# ! Don't add more, other tests assume 4 entries
DIMS_TIPS = [
    dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
    dict(y="tip", x="size-cut", hue="smoker", col="sex"),
    dict(y="tip", x="size-cut", hue="smoker"),
    dict(y="tip", x="size-cut"),
]

DIMS_FMRI = [
    dict(y="signal", x="timepoint", hue="event", col="region"),
    dict(y="signal", x="timepoint", hue="region", col="event"),
    dict(y="signal", x="timepoint", hue="region"),
    dict(y="signal", x="timepoint", hue="event"),
    dict(y="signal", x="timepoint"),
]


#%% 

