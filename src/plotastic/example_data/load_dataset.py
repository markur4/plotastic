import pandas as pd
import pkg_resources
import os

### List all available datasets


FILES = dict(
    fmri="fmri.xlsx",  # * Removed timepoints bigger than 10
    tips="tips.xlsx",  # * Added a size-cut column pd.cut(df["size"], bins=[0, 2, 10], labels=["1-2", ">=3"])
)

DIMS = dict(
    fmri=dict(y="signal", x="timepoint", hue="event", col="region"),
    tips=dict(y="tip", x="size-cut", hue="smoker", col="sex", row="time"),
)


def _get_dataframe_from_package(name: str = "tips"):
    """Retrieves Dataframe from package data directory

    :param name: _description_, defaults to "tips"
    :type name: str, optional
    :return: _description_
    :rtype: _type_
    """

    ### Get the path to the package data directory for 'example_data'
    package = "plotastic.example_data"  # * Needs to be importable
    relative_path_to_file = os.path.join("data", FILES[name])
    file_path = pkg_resources.resource_filename(package, relative_path_to_file)

    return pd.read_excel(file_path)

    # ### List all XLSX files in the 'data' directory
    # xlsx_files = [f for f in os.listdir(package_data_dir) if f.endswith('.xlsx')]

    # ### Read the XLSX file into DataFrames
    # for file_name in xlsx_files:
    #     key = file_name.split('.')[0] # * Remove the xlsx extension
    #     if key == name:
    #         #* Get the full path to the file
    #         file_path = os.path.join(package_data_dir, file_name)
    #         #* import the file as a DataFrame
    #         return pd.read_excel(file_path)


# FILES = dict(
#     fmri="fmri.xlsx",  # * Removed timepoints bigger than 10
#     tips="tips.xlsx",  # * Added a size-cut column pd.cut(df["size"], bins=[0, 2, 10], labels=["1-2", ">=3"])
# )


def load_dataset(name: str = "tips", verbose=True) -> tuple[pd.DataFrame, dict]:
    """
    Executes seaborn.load_dataset, but also:
    - Assigns column names to ["y","x","hue","col","row",] in a dictionary called dims
    - Converts ["x","hue","col","row"] into ordered categorical datatype

    :param verbose: whether to print out assigned dims dictionary
    :param name: Name of the dataset. ["fmri", "tips"]
    :return: Example data and dictionary for dimensions
    :rtype: tuple[pd.DataFrame, dict]
    """

    assert name in FILES, f"#'{name}' should have been one of {list(FILES.keys())}"

    ### Import DataFrame from package
    package = "plotastic.example_data"  # * Needs to be importable
    relative_path_to_file = os.path.join("data", FILES[name])
    file_path = pkg_resources.resource_filename(package, relative_path_to_file)
    df = pd.read_excel(file_path)
    
    ### Get dims
    dims = DIMS[name]

    # df = sns.load_dataset(name)

    # keys = [
    #     "y",
    #     "x",
    #     "hue",
    #     "col",
    #     "row",
    # ]

    # ### DEFINE FACTORS
    # if name == "tips":
    #     factors = ["tip", "size-cut", "smoker", "sex", "time"]
    #     df["size-cut"] = pd.cut(df["size"], bins=[0, 2, 10], labels=["1-2", ">=3"])
    # elif name == "fmri":
    #     factors = ["signal", "timepoint", "event", "region"]
    #     df = df[df["timepoint"] < 10]

    # ### MAKE CATEGORICAL
    # for col in factors[1:]:  # don't include y
    #     df[col] = pd.Categorical(df[col], ordered=True)

    # if name == "fmri":
    #     # df = df.where(df["timepoint"] < 10) # make it shorter
    #     df["timepoint"] = df["timepoint"].cat.remove_unused_categories()

    # _dims = dict(zip(keys, factors))

    if verbose:
        print(
            f"#! Imported seaborn dataset '{name}' \n\t columns:{df.columns}\n\t dimensions: {dims}"
        )

    return df, dims
