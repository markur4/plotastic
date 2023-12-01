import pandas as pd
import pkg_resources
import os

### List all available datasets

FILES = dict(
    fmri="fmri.xlsx",  #' Removed timepoints bigger than 10
    tips="tips.xlsx",  #' Added a size-cut column pd.cut(df["size"], bins=[0, 2, 10], labels=["1-2", ">=3"])
    qpcr="qpcr.xlsx",
)

DIMS = dict(
    fmri=dict(y="signal", x="timepoint", hue="event", col="region"),
    tips=dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
    qpcr=dict(y="fc", x="gene", hue="fraction", col="method", row="class"),
)


def load_dataset(name: str = "tips", verbose=True) -> tuple[pd.DataFrame, dict]:
    """Executes seaborn.load_dataset, but also returns dictionary that assigns dimensions
    to column names ["y","x","hue","col","row"]

    :param verbose: Prints information and dims dictionary
    :param name: Name of the dataset. Error messayge contains available options. Defaults to "tips"
    :return: Example data and dictionary for dimensions
    :rtype: tuple[pd.DataFrame, dict]
    """

    ### Check user Arguments
    assert (
        name in FILES
    ), f" '{name}' should have been one of {list(FILES.keys())}"

    ### Import DataFrame from package
    package = "plotastic.example_data"  #' Needs to be importable
    path_relative = os.path.join(
        "data", FILES[name]
    )  #' Path with python package as root
    path_full = pkg_resources.resource_filename(package, path_relative)
    df = pd.read_excel(path_full)

    ### Get dims
    dims = DIMS[name]

    if verbose:
        print(
            f"#! Imported seaborn dataset '{name}' \n\t columns:{df.columns}\n\t dimensions: {dims}"
        )

    return df, dims
