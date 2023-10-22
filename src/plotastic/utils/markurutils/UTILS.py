from __future__ import annotations
from functools import wraps
from json import load

from typing import (
    Hashable,
    Collection,
    Any,
    Callable,
    TYPE_CHECKING,
)  # , List, Optional, Set, Tuple, Union
from datetime import datetime, date
import time


import pickle
import io
import shutil

import gc
import sys
import os
import os.path
from pathlib import Path

from joblib import Parallel, delayed

import ipynbname
from IPython import get_ipython

from IPython.display import display, HTML, DisplayObject
import warnings

import numpy as np
import pandas as pd

# import pingouin as pg
# from statsmodels.stats import anova
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

# pd.options.mode.chained_assignment = None  # default='warn'
# pd.set_option("display.precision", 3)

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.font_manager as font_manager


if TYPE_CHECKING:
    import pyensembl



""" 3 Time >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' 3 Time  ">


def get_day():
    return date.today().strftime("%Y%m%d")


def get_time():
    return datetime.now().strftime("%H%M%S")


def timeit_verbose(func):
    """decorator that measures time of a function to complete"""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def timeit(func):
    """decorator that measures time of a function to complete"""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(
            f"Function {func.__name__} ({len(args)} args, {len(kwargs)} kwargs), Took {total_time:.4f} seconds"
        )
        return result

    return timeit_wrapper


# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


""" I/O Paths & Files >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' I/O Paths & File  ">

def get_terminal_width():
    try:
        # Get the terminal width
        columns, _ = shutil.get_terminal_size(fallback=(80, 24))
        return columns
    except Exception:
        # Handle exceptions if the terminal size cannot be determined
        return 80  # Default value
    
def print_separator(char='='):
    # Get the terminal width
    terminal_width = get_terminal_width()
    
    # Calculate the number of characters required
    num_chars = terminal_width // len(char)
    
    # Print the separator line
    print(char * num_chars)

def glob_searchfilename(path: "Path", filename: str, rettype="list"):
    rettypes = ["list", "str"]
    assert (
        rettype in rettypes
    ), f"#! rettype {rettype} should have been one of {rettypes}"

    rough_matches = path.glob(f"*{filename}*")
    files = [
        str(path.stem) for path in list(rough_matches)
    ]  # Find all pdf files in outpath

    """Making one big string saves us the for loop for re searches. Needs flag re.MULTILINE"""
    if rettype == "list":
        return files
    elif rettype == "str":
        return "\n".join(files)


def insert_subfolder(
    filepath: str | "Path",
    subfoldername: str = None,
    prefix="",
    undo_overwrite_protection=False,
) -> "Path":
    """
    Takes a filepath and inserts a subfolder before the filename (for e.g. multiple exports of one dataset per timepoint.
    :param prefix: str. Will be added in front of the new subfolder
    :param filepath: filepath ('/user/experiment/bla1')
    :param undo_overwrite_protection:
    :return:         filepath ('/user/experiment/bla1/bla1')
    """

    """CAST PATH and REMOVE FILE EXTENSION"""
    filepath = Path(filepath)

    filename = filepath.name
    filestem = filepath.stem
    parent = Path(filepath).with_suffix("").parent
    if subfoldername:
        filestem = subfoldername

    # print(filepath)

    """(filename ends with _1 due to Filer)"""
    if undo_overwrite_protection:
        print(undo_overwrite_protection)
        end_of_name = filepath.stem.split("_")[-1]
        try:
            int(end_of_name)
        except ValueError:
            ends_with_number = False
        else:
            ends_with_number = True

        """OVERRIDE FILESTEM"""
        if ends_with_number:
            print(ends_with_number)
            steml = filestem.split("_")[
                :-1
            ]  # stem IS FILENAME WITHOUT FILE EXTENSION SUFFIX
            filestem = ("_").join(steml)

        else:
            warnings.warn(
                f"#! undo_overwrite_protection is {undo_overwrite_protection}, although file does not end with a number"
            )
            filestem = filestem

    """CONSTRUCT NEW FILEPATH"""
    # name = Path(prefix + filepath.stem)  # GET NAME OF THE FILE
    # subfolder = filepath

    subfolder = parent / filestem
    subfolder.mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(subfolder):
    #     os.mkdir(subfolder)
    filepath_new = parent / filestem / filename
    return filepath_new


def is_notebook() -> bool:
    """checks from where the script is executed. Handy if you want to print() or use HTML based outputs"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_scriptname() -> str:
    if is_notebook():
        try:
            scriptname = ipynbname.name()
        except FileNotFoundError:
            scriptname = "Interactive"
    else:
        scriptname = Path(
            sys.argv[0]
        ).stem  # os.path.basename(sys.argv[0]) #Path().name
    return scriptname


def exec_pys(pys: Collection) -> None:
    """Takes a list with filepaths to .py scripts and executes these scripts"""
    # from importlib.machinery import SourceFileLoader

    startcwd = os.getcwd()

    """Print out the names and paths of the .py scripts"""
    print("\n||======================= TESTIN' DEESE PIES: =======================")
    for i, scriptpath in enumerate(pys):
        name = Path(scriptpath).stem
        print(f"||\t{i}. ")
        print(f"||\t NAME:      '{name}' ")
        print(f"||\t LOCATION:  '{scriptpath}' \n|| ")
    print("||==================================================================")

    """Execute the scripts"""
    for scriptpath in pys:
        name = Path(scriptpath).stem
        print(
            "\n\n|| New .py script starting: ==============================================================================="
        )
        print(f"||   EXECUTING:  '{name}' ")
        print(f"||   LOCATION:  '{scriptpath}' ")

        """Change working directory so all the output is saved within script location"""
        os.chdir(scriptpath.parent)
        os.system(f"python '{name}.py'")

        """Reset cwd"""
        os.chdir(startcwd)
        # SourceFileLoader(name, path=name+".py").load_module() #import module by filename


# </editor-fold"> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


""" I/O Print-outs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' I/O Print-outs  ">

from contextlib import (
    contextmanager,
)  # used to suppress output using < with mku.suppress_stdout(): >

# * Ignore warning messages using   <warnings.filterwarnings('ignore') >  and < warnings.resetwarnings() >


@contextmanager
def suppress_stdout() -> None:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def ignore_warnings(func) -> callable:
    """DECORATOR that ignores warnings"""

    # @timeit
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

        # warnings.filterwarnings('ignore')
        # return func(*args, **kwargs)
        # warnings.resetwarnings()

    return wrapper


# </editor-fold"> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


""" I/O Objects"""


def copy_by_pickling(obj, plt_close=True):
    """converts to bytes and loads it to return it"""

    # * CONVERT TO BYTE RAM
    # with io.BytesIO() as buf: # ! 'with' statement not working with pyplot
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)

    # * RELOAD IT
    copy = pickle.load(buf)
    # p2 = pickle.loads(buf.getvalue()) # THIS GETS VALUE WITHOUT RESETTING buf

    # * MPL IS ANNOYING
    # if plt_close:
    plt.close()

    return copy


""" ENSEMBL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""


def get_ens_from_name(gene_symbol: str, genome: "pyensembl.EnsemblRelease" = None):
    if not genome:
        genome = pyensembl.EnsemblRelease(species="human")

    """SEARCH DATABASE FOR GENE NAME"""
    try:
        matchlist = genome.genes_by_name(gene_symbol)
    except ValueError:
        """NO RESULTS FOUND FOR QUERY"""
        return "No Result"  # [float("NaN")]

    """ONE GENE NAME MIGHT YIELD MULTIPLE RESULTS"""
    ENS_list = []
    for match in matchlist:
        ENS_list.append(match.gene_id)

    if len(ENS_list) == 1:
        return ENS_list[0]
    else:
        return ENS_list


""" Parallelize"""


def runparallel(func: "Callable", inputs, funckwargs: dict, n_jobs=4):
    """

    :param func: Callable
    :param inputs: dict,
    | - e.g. {key: input, key2: input2}
    | - e.g. pd.DataFrame.groupby(by)
    :param funckwargs:
    :param n_jobs:
    :return: dict, {key: result, key2: result2}
    """

    R = Parallel(n_jobs=n_jobs)(delayed(func)(**funckwargs) for key, inp in inputs)
    RDict = {key: result for key, result in R}

    return RDict


### Try to parralelize
# R = Parallel(n_jobs=2)(
#     delayed(self._base_pairedttests)(name=name, data=df,
#                                      w1=w1, w2=w2,
#                                      **standard_kws, **kwargs)
#     for name, df in self.data.groupby(by)
# )
# DFdict = {name:result for name,result in R}
# DFdict = mku.runparallel(func = self._base_pairedttests,
#                          inputs = self.data.groupby(by),
#                          funckwargs = dict(data=df, w1=w1, w2=w2, **standard_kws, **kwargs),
#                          )

""" 5 Numerical >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' 5 Numerical  ">

# from scipy.stats import norm
# from statannot import add_stat_annotation


def cont_distribution(series, bins=200):
    """
    Uses Kernel Density estimation to retrieve a probability density function (PDF). The PDF is a Kernel (Faltungsmatrix), that can be fed with an array of x-values to return an array of estimated y values that represent the probability of retrieving a y-value at a given X. The integral of the PDF is 1 (makes sense, probabilty of getting all values is 100%)

    :param series:  a List like structure
    :param bins: number of bins, the bigger the more accurate
    :return: x and y values of a continuous distribution. Yscale
    """
    ## Generate A KERNEL that describes the PDF (Probability Density Function)
    kernel = gaussian_kde(series, bw_method="silverman")
    ### Retrieve an X-Axis with equal stepsize to use as an input for the Kernel
    ### dx is the steplength in raw units #dx = x[1] - x[0]
    x, dx = np.linspace(series.min(), series.max(), num=bins, retstep=True)

    # Retrieve y values (kernel.evaluate()) given an array
    yDensity = kernel(x)
    yCounts = kernel(x) * len(series) * dx
    return x, yDensity, yCounts  # series


def histogram(series, bins=200):
    """All but the last (righthand-most) bin is half-open. In other words, if bins is: [1, 2, 3, 4] then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4."""
    x, dx = np.linspace(series.min(), series.max(), num=bins, retstep=True)
    yCounts, rightedges = np.histogram(series, bins=x)
    # print(counts, rightedges)
    ### Counts has 1 point less than x. remove last point
    x = x[:-1]
    ### Get Density
    yDensity = yCounts / len(series) / dx

    return x, yDensity, yCounts  # series


def mean_over_range(
    x,
    y,
    xRange=None,
):
    """Takes Y-Values over a certain x range and returns mean"""
    if xRange:
        x = np.array([X if xRange[0] < X < xRange[1] else None for X in x])
        y = y[x != np.array(None)]
    return y.mean()


def getpeak(x, y, xRange=None, type="maximum"):
    """finds a peak within range. Requires scipy.signal.find_peaks"""
    if xRange:
        x = np.array([X if xRange[0] < X < xRange[1] else 0 for X in x])
        y = np.array([Y if X != 0 else 0 for X, Y in zip(x, y)])

    # peakIndex = find_peaks(y, height=0.000010)
    # if type=="maximum":
    #     c = np.greater
    # elif type=="minimum":
    #     c = np.less
    # peakIndices = argrelextrema(y ,comparator=c, order=20)

    if type == "minimum":
        peakIndices, props = find_peaks(-y)
    elif type == "maximum":
        peakIndices, props = find_peaks(y)

    # peakIndices = peakIndices[0] # is made for multidimensional arrays
    yList = [y[i] for i in peakIndices]
    xList = [x[i] for i in peakIndices]
    xList = list(xList)
    if type == "minimum":
        peakY = np.amin(yList)
    if type == "maximum":
        peakY = np.amax(yList)
    peakX = xList[yList.index(peakY)]

    # if xRange:
    #     x = [X if xRange[0] < X < xRange[1] else float("NaN")  for X in x]
    #     y = [Y if X!=float("NaN")     else float("NaN")  for X,Y in zip(x,y)]
    #
    # if type=="maximum":
    #     peakY = max(y)
    # if type=="minimum":
    #     peakY = min(y)
    #
    # peakX = x[y.index(peakY)]
    # print(peakX,peakY)
    return peakX, peakY


# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


# == 6 Pandas >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' 6 Pandas ">
"""EXAMPLE DATA"""


def load_dataset(name="tips", verbose=True):
    """
    Executes seaborn.load_dataset, but also:
    - Assigns column names to ["y","x","hue","col","row",] in a dictionary called dims
    - Converts ["x","hue","col","row"] into ordered categorical datatype

    :param showdims: whether to print out assigned dims dictionary
    :param name: Name of the dataset. ["fmri", "tips"]
    :return: df:pd.DataFrame, dims:dict
    """

    names = ["fmri", "tips"]
    assert name in names, f"#mk! '{name}' should have been one of {names}"

    df = sns.load_dataset(name)

    keys = [
        "y",
        "x",
        "hue",
        "col",
        "row",
    ]

    """DEFINE FACTORS"""
    if name == "tips":
        factors = ["tip", "size-cut", "smoker", "sex", "time"]
        df["size-cut"] = pd.cut(df["size"], bins=[0, 2, 10], labels=["1-2", ">=3"])
    elif name == "fmri":
        factors = ["signal", "timepoint", "event", "region"]
        df = df[df["timepoint"] < 10]

    """MAKE CATEGORICAL"""
    for col in factors[1:]:  # don't include y
        df[col] = pd.Categorical(df[col], ordered=True)

    if name == "fmri":
        # df = df.where(df["timepoint"] < 10) # make it shorter
        df["timepoint"] = df["timepoint"].cat.remove_unused_categories()

    _dims = dict(zip(keys, factors))
    if verbose:
        print(
            f"#! Imported seaborn dataset '{name}' \n\t columns:{df.columns}\n\t dimensions: {_dims}"
        )
    return df, _dims


def get_testdata(name="tips"):
    """Alias for load_dataset"""
    df, dims = load_dataset(name=name, verbose=True)
    return df, dims


"""GET DF INFO"""


def df_info(df):
    print(f"Total Memory Usage: {df.memory_usage(deep=True).sum() / 10e6} MB")
    infoDF = pd.concat(
        objs=[
            df.dtypes,
            df.memory_usage(deep=True) / 10e6,
        ],
        keys=["dTypes", "RAM [MB]"],
        axis=1,
    ).transpose()
    pretty_print(infoDF)


def characterize_DF(DF):
    h = 5
    w = (h * 0.35) * len(DF.columns)
    g = DF.plot(kind="box", subplots=True, figsize=(w, h), rot=90)
    plt.subplots_adjust(wspace=0.6)
    for i, ax in g.items():
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
    plt.show()


def drop_columns_by_regex(DF: "pd.DataFrame", pattern: str):
    DF = DF[DF.columns.drop(list(DF.filter(regex=pattern)))]
    return DF


"""PRINT"""


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def pp(df, prec: int = None, ret: bool = False) -> "DisplayObject":
    """
    Displays pandas dataframe to jupyter notebook
    :param df:
    :param prec: display precision of floats
    :param ret:  If True (default), also returns the dataframe
    :return:
    """
    if prec:
        # with pd.option_context('display.float_format', '${:,.2f}'.format):
        with pd.option_context("display.precision", prec):
            display(HTML(df.to_html().replace("\\n", "<br>")))
    else:
        display(HTML(df.to_html().replace("\\n", "<br>")))

    if ret:
        return df


"""MANIPULATE DF"""


def catchstate(df, var_name: str = "df"):
    """
    Helper function that captures intermediate Dataframes mid-chain.
    In the global namespace, make a new variable called var_name and set it to dataframe
    :param df: Pandas dataframe
    :param var_name:
    :return:
    """
    globals()[var_name] = df
    return df


def flatten_cols(DF):
    DF.columns = DF.columns.map(" | ".join).str.strip(" | ")
    return DF


def unflatten_cols(DF, at: str = " | ", cols: list | tuple = None):
    """
    Splits columns at a string and expands resulting colnames into multicolumns.
    Helpful if one column name contains multiple facets.
    :param DF:
    :param at:
    :param cols:
    :return:
    """

    if not cols:
        DF.columns = DF.columns.str.split(at, expand=True)
        DF.sort_index(axis=1, inplace=True)
    else:
        DF.set_index([c for c in DF.columns if not c in cols], inplace=True)
        DF.columns = DF.columns.str.split(at, expand=True)
        """DON'T RESET INDEX NOW, IF YOU WANT TO STACK"""
        # DF.reset_index(inplace=True)
    """REVOME LEADING AND TRAILING WHITESPACES"""
    DF.rename(str.strip, axis="columns", inplace=True)

    return DF


def multi_categorical(df, catdict, renamedict=None, verbose=True):
    """
    :param df:
    :param catdict:     catdict = {
                            'MSC : INA-6':    ["1:1", "1:2", "1:4"],
                            'INA-6–Count per MSC': ['0', '≥ 1', '≥ 3', '≥ 6'],
                            'Patient':     ['1571', '1573', '1578'],
                            'Co-Culture Duration [h]': [24,48],
                                }
    :param renamedict: renamedict = {'Co-Culture Duration [h]': ["24 h","48 h"], }
    :return:
    """

    for col, lvls in catdict.items():
        oldType = type(df[col].tolist()[0])
        newType = type(lvls[0])
        if oldType != newType:
            if verbose:
                print(
                    f"#! '{col}' has type ({oldType}), but {newType} was passed! Converting '{col}' into {newType}"
                )
            df[col] = df[col].astype(newType)
            # print(df.dtypes)

        ### REMOVE UNSEEN LEADING AND TRAILING SPACES FROM DF TO PREVENT FRUSTRATION
        if df[col].dtype is str:
            df.rename(str.strip, axis="columns", inplace=True)  ## COLUMN NAMES
            df[col] = df[col].str.strip()  ## COLUMN LEVELS

        ### CONVERT TO CATEGORICAL"""
        df[col] = pd.Categorical(df[col], categories=lvls, ordered=True)

    if renamedict:
        for col, names in renamedict.items():
            if col in catdict:
                df[col] = df[col].cat.rename_categories(names)

    return df


def melter(df, cols: list | str = None, varname="variable", valname="value", ret=True):
    """
    TURNS A COLUMN INDEXER INTO A ROW INDEXER.
    Converts several Columns of the same datatype into one column (For long-format!).
    Function to Melt or "Unpivot"
    Melting takes a column and 'downgrades' it to a level of a new column:
    1. Take column name
    2. Make a new column where each row contains the column name (variable)
    3. Make a second new column that contains the columns values

    Melting just one column (`len(value_vars) = 1`) does not gain anything.
    Melting two columsn however, allows us to add another index/category level to each value.
    This makes only sense if both columns contain values of similar scale.
    E.g. melting cols ("Size_Female", "Size_Male") makes sense, since "Size" belongs into one column
    E.g. melting cols ("Size_Female", "Mass_Male") does not make sense, since "Size" and "Weight" should be two differen columns

    This Melting function that does not change Dataframe shape.
    Pandas usually takes two arguments.
    But we only need one

    :param list:
    :param df: Dataframe
    :param cols: Columns to melt
    :param valname:
    :param varname:
    :return: dataframe
    """
    if not cols:
        cols = df.columns

    """NON-SELECTED COLUMNS ARE PASSED TO id_vars TO MAINTAIN DF-SHAPE """
    df_columns = list(df.columns)
    for c in cols:
        df_columns.remove(c)
    dfOut = df.melt(
        id_vars=df_columns,  # COLUMNS THAT SERVE AS "INDEX"
        value_vars=cols,  # COLUMNS WHOSE COLUMN NAMES WILL BE DOWNGRADED TO LEVELS
        var_name=varname,  #
        value_name=valname,
    )
    dfOut = dfOut.reset_index()
    del dfOut["index"]
    if ret:
        return dfOut


"""APPEND"""


def appendrow(df, List):
    df.loc[len(df)] = List
    return df


def insert_after(
    df,
    after: str,
    newcol: list | pd.Series = None,
    newcol_name=None,
    func: callable = None,
) -> pd.DataFrame:
    """
    inserts a column after specified column. Changes dataframe inplace!
    :param df: pandas dataframe
    :param after: Column name to insert after
    :param newcol: Data seried
    :param newcol_name:
    :param func:
    :return: df
    """
    if newcol_name:
        newcol_name = newcol_name
    elif func:
        newcol_name = after + func.__name__
    else:
        newcol_name = after + "_0"

    if newcol:
        newcol = newcol
    elif func:
        newcol = func(df[after])
    else:
        raise Exception("#! insert_after: Must pass either func or newcol")

    location = df.columns.get_loc(after)
    df.insert(loc=location + 1, column=newcol_name, value=newcol)

    return df


"""CALCULATIONS"""


def groupcalc(
    df,
    func: str | Callable,
    by: str | list,
    col: str | list[str],
    where: "pd.Series" = None,
    # `where= df["rank"] == 1`
    aggfunc: str = "mean",
    verbose=True,
) -> "pd.Series" | "pd.DataFrame":
    """
    Performs a mathematical operation per group using EITHER one value that's unique within that group OR an
    aggregated value (mean, min, max, any, all) from every value of the group.

    STRATEGY:
    |- Make an operand with only by as index and values that are to be added to/subtracted from df[col]
    |- ADD "by"-columns as another index level to an intermediate df_i
    |- The operand will apply each number to df_i per group accordingly, since only by index level(s) match
    |- remove 'by' from result index
    |- return result, which can be added to df

    :param df: pd.DataFrame
    :param func: str. Name of a method [] from pd.DataFrame or pd.Series that performs a mathematical operation
    :param by: str, The column to group by
    :param col: str. Column name to operate on
    :param where: an indexer like. e.g. `where= df["rank"] == 1`. That row is taken as an operand to operate on col.
    We assume that "where" is an indexer that yields a single row when grouping the dataframe
    :param agg: str, function name that aggregates multiple values from each group.
    :param verbose: If True, prints the operand (=the amounts that will e.g. subtracted from the dataframe)
    :return:
    """

    """MAKE AN OPERAND WHOSE ONLY INDEX IS GROUPBY"""
    if not where is None:  ## BY A SPECIFIC ROW
        operand = (
            df.loc[where]
            .reset_index()
            .set_index(by)[  # <-- EXCLUDE EVERY INDEX EXCEPT GROUPBY INDEX
                col
            ]  ## DON'T RETURN 1 DIMENSIONAL DATAFRAME
        )
    elif not aggfunc is None:  ## GET VALUE BY AGGREGATING
        bycol = [by, col] if isinstance(col, str) else [by] + [c for c in col]
        print(bycol)
        operand = (
            df.reset_index()[bycol]
            .set_index(by)  # <-- EXCLUDE EVERY INDEX EXCEPT GROUPBY INDEX
            .groupby(by)
            .agg(aggfunc)[col]  ## DON'T RETURN 1 DIMENSIONAL DATAFRAME
        )
    else:
        raise ValueError("#! Must pass either where or aggfunc")

    print(operand)

    """INCLUDE GROUPBY COLUMN INTO INDEX"""
    df_i = df.set_index(by, append=True)

    """IF func IS A STRING, WE EXPECT A pd.DataFrame.method"""
    """IF func IS CALLABLE, WE EXPECT A NUMPY FUNCTION"""
    if isinstance(func, str):
        result = getattr(df_i[col], func)(operand)  # EQUALS: df[x].subtract(subtrahend)
    elif callable(func):
        if isinstance(operand, pd.Series):
            result = func(df_i[col], operand)
        else:  ## OPERAND IS A DATAFRAME IF MULTIPLE COLS ARE PASSED
            results = []
            for c in operand.columns:
                results.append(func(df_i[c], operand[c]))
            result = pd.concat(results, axis=1)
    else:
        raise ValueError(f"#! {func} should have been of type str or callable")

    """REMOVE THE GROUPBY INDEX TO RESTORE ORIGINAL INDEX STRUCTUE"""
    result.reset_index(
        level=by, drop=True, inplace=True  # only removes the given level
    )

    """PRINT RESULT"""
    if verbose:
        print(f"#! {func} by these values:")
        if len(operand) > 10:
            print("Unique valuecounts of operand mask")
            # print(operand)
            print(operand.value_counts())
        else:
            print(operand)

    """TESTING"""

    def test_groupcalc_where():
        df2 = df.copy().set_index("CarNo")
        # mku.pp(df2)
        df2["LapTime_diff"] = groupcalc(
            df2,
            func="subtract",
            by="UpdateTime",
            col="LapTime",
            where=df2["rank"] == 1,
        )
        pp(df2)

        #
        ## ALSO WORKS WITH NUMPY FUNCTIONS

        df2["LapTime_diff"] = groupcalc(
            df2,
            func=np.subtract,  # don't confuse np.add with np.sum
            by="UpdateTime",
            col="LapTime",
            where=df2["rank"] == 1,
        )
        pp(df2)

        ## ALSO WORKS WITH NUMPY FUNCTIONS
        df2[["LapTime_diff", "LAPS_diff"]] = groupcalc(
            df2,
            func=np.subtract,  # don't confuse np.add with np.sum
            by="UpdateTime",
            col=["LapTime", "LAPS"],
            where=df2["rank"] == 1,
        )
        pp(df2)

        ## ALSO WORKS WITH MULTIPLE COLUMNS
        df2[["LapTime_div", "LAPS_diff"]] = groupcalc(
            df2,
            func="subtract",
            by="UpdateTime",
            col=["LapTime", "LAPS"],
            where=df2["rank"] == 1,
        )
        pp(df2)

    test_groupcalc_where()

    def test_groupcalc_agg():
        df2 = df.copy().set_index("CarNo")
        # mku.pp(df2)
        df2["LapTime_norm"] = groupcalc(
            df2, func="divide", by="UpdateTime", col="LapTime", agg="max"
        )
        pp(df2)

        #
        ## ALSO WORKS WITH NUMPY FUNCTIONS

        df2["LapTime_norm"] = groupcalc(
            df2,
            func=np.divide,  # don't confuse np.add with np.sum
            by="UpdateTime",
            col="LapTime",
            agg="max",
        )
        pp(df2)

        ## ALSO WORKS WITH NUMPY FUNCTIONS
        df2[["LapTime_norm", "LAPS_norm"]] = groupcalc(
            df2,
            func=np.divide,  # don't confuse np.add with np.sum
            by="UpdateTime",
            col=["LapTime", "LAPS"],
            agg="max",
        )
        pp(df2)

        ## ALSO WORKS WITH MULTIPLE COLUMNS
        df2[["LapTime_norm", "LAPS_norm"]] = groupcalc(
            df2, func="divide", by="UpdateTime", col=["LapTime", "LAPS"], agg="max"
        )
        pp(df2)

    test_groupcalc_agg()

    return result


# <editor-fold desc="7 << Well Coordinate Management  > > > > > > > > > > > > > > > > ">


def wellParams(row, wellCol: str, plateCol: str, wellDict):
    """
    function to be used with df.apply().
    :param row:
    :param wellCol:
    :param plateCol:
    :param wellDict:
    :return:
    """
    # global Rename
    # DFwell = row[Rename[Wells]]
    # plate = row[Rename[Plate]]
    DFwell = row[wellCol]
    plate = row[plateCol]
    notfound = []
    for dims, Dictwells in wellDict.items():
        if DFwell in Dictwells:
            ##: If plate number is specified:
            if plate == Dictwells[0] or type(Dictwells[0]) == str:
                return dims
    ##: When loop ends without match, well wasn't specified in WellDict
    return "unspecified"


def reportDF(DF, wellcol: str = "well"):
    print(f"1 Raw DF has shape {DF.shape}")

    unspecified = DF[DF["Well Content"] == "unspecified"][wellcol].to_list()
    specified = DF[DF["Well Content"] != "unspecified"][wellcol].to_list()
    unspecified_list = sorted(list(set(unspecified)))
    specified_list = sorted(list(set(specified)))
    if len(unspecified_list) == 0:
        print(
            f"2 All wells are specified by WellDic (total of {len(specified_list)} wells)"
        )
    else:
        print(
            f"2 !!!! Specified {len(specified_list)} wells, but these wells were left unspecified by WellDict: \n{unspecified_list}"
        )

    nanSet = set()
    DF[DF.isna().any(axis=1)]

    print(f"### Detected These Rows with NaN:")


def ignoreWells(df, ignoreDict):
    outDF = df.copy(deep=True)
    oldshape = outDF.shape

    indices = []
    delDFs = []
    for plate, wells in ignoreDict.items():
        for well in wells:
            # pretty_print(outDF.head())
            delRow = outDF[(outDF["Plate"] == plate) & (outDF["Well"] == well)]
            index = delRow.index
            outDF = outDF.drop(index=index)

            indices.append(index)
            delDFs.append(delRow)

    newshape = outDF.shape
    print(
        f"!!! !DELETED Specific wells from Plates: \n\tBefore:\t{oldshape}\n\tAfter:\t{newshape}"
    )
    if len(delDFs) > 0:
        delDF = pd.concat(delDFs)
        pretty_print(delDF.head(5))

    return outDF


def catsplitter(DF, col: str, dims: dict, char="|", report=False):
    """
    Splits columns by " | " into more columns, renames these columns and sets them as categorical
    INPUT:
    - Dataframe
    - Col: Column in the Dataframe that contain the Categories of the Plate Dimensions which are separated by |
    - Dim: Dim The Names of the Dimensions. These are the names of the new splitted columns
        Dims = {
                    "Time":     ["t8", "t9", "t10"],
                    "Fraction": ["V2", "V2-Removed", "V3"],
                }
    OUTPUT: Modified Dataframe
    """
    # copy the dataframe
    outDF = DF.copy(deep=True)
    # gather the Names of the dataframe columns that contain Dimensions separated by |
    Dim_Names = [e for e in dims.keys()]
    Dim_Categories = [e for e in dims.values()]
    # Split the columns into dimensions
    outDF[Dim_Names] = outDF[col].str.split(char, expand=True)
    # pretty_print(outDF.head(5))

    # delete the old columns
    del outDF[col]
    # Make these columns categorical
    for dim, categories in dims.items():
        outDF[dim] = outDF[
            dim
        ].str.strip()  # remove leading and following spaces in each column that come with
        # pretty_print(outDF.head(5))
        outDF[dim] = pd.Categorical(outDF[dim], categories=categories, ordered=True)
        # pretty_print(outDF.head(5))
    # review output
    if report == True:
        pretty_print(outDF.head(5))
        pretty_print(outDF.tail(5))
    # pretty_print(DF)
    # pretty_print(df[df["Well"]=="A1"].head(5))
    return outDF


# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


def countHist(g, display="table", colors=None, fontsize=25):
    """sums up all heights in an histogram and adds number into legend"""
    if not colors:
        colors = sns.color_palette()

    ### Iterate through axes:
    for ax in g.axes.flat:
        ### Initialize dict:
        Sums = {to_hex(p.get_facecolor()): 0 for p in ax.patches}
        ### Iterate through bars and sum up heights
        for patch in ax.patches:
            color = to_hex(patch.get_facecolor())
            y = patch.get_height()
            Sums[color] += y
        Sums = {color: "Σ: {:.1e}".format(int(Sum)) for color, Sum in Sums.items()}

        if display == "table":
            tableDf = pd.DataFrame.from_dict(Sums, orient="index").reset_index()
            del tableDf["index"]
            # pretty_print(tableDf)
            T = ax.table(
                cellText=tableDf.values,
                rowLabels=None,
                colLabels=None,
                cellLoc="right",
                rowLoc="center",
                loc="right",
                zorder=100,
                edges="open",
                bbox=[
                    0.55,
                    1.0 - (0.1 * len(Sums)),
                    0.5,
                    0.1 * len(Sums),
                ],  # [left,bottom,width,height]
            )
            T.auto_set_font_size(False)
            T.set_fontsize(fontsize)
            for i, color in enumerate(Sums):
                T[(i, 0)].set(fill=False)
                T[(i, 0)].set_text_props(color=color, fontstretch="condensed")


def diluteHist(g, dilutionDict, sharey=False):
    """takes a dictionary and changes"""

    global Switch

    for ax in g.axes.flat:
        # capture the title of the axis
        row = ax.get_title().split(" | ")[0].split(" = ")[1]
        col = ax.get_title().split(" | ")[1].split(" = ")[1]
        if Switch == True:
            row_col = f"{col} | {row}"
        else:
            row_col = f"{row} | {col}"  # << that's how it's supposed to be

        for key, value in dilutionDict.items():
            # print(key)
            if key == row_col:
                print(f"{row_col}: Applying factor {'{:.2f}'.format(value)}")
                Y = []
                for patch in ax.patches:
                    y = patch.get_height()
                    ydil = y * value
                    patch.set_height(ydil)
                    Y.append(ydil)
                if len(Y) > 0:
                    if not sharey:
                        ax.set(ylim=(0, max(Y)))
                    else:
                        plt.ylim = (0, max(Y))


# def group_enumerate(DF, by, colname="Gr.Index"):
#     """
#     Groups the dataframe by a column name (a variable or factor), enumerates these groups and adds the index of that group as a new column to that Dataframe
#     Each factor has multiple replicates.
#     Group by each combination of factors to yield groups containing each replicate
#     """
#     groups = DF.groupby(by)
#     DFdict = {}
#     for i, (name, df) in enumerate(groups):
#         DFdict[i] = df
#     outDF = pd.concat(DFdict)
#
#     outDF = outDF.reset_index()
#     del outDF["level_1"]
#     outDF = outDF.rename(columns={"level_0": colname})
#
#     return outDF

# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


""" 7 Matplotlib & Fonts >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""


# <editor-fold desc="''' 7 Matplotlib & Fonts   ">
# import gc
#
# @timeit
# def clear_ram():
#     #plt.gcf()
#     mklab_Garbage = gc.get_objects()
#     print(mklab_Garbage)
#
#     plt.close()
#     refs = gc.get_referrers()
#
#     gc.collect()
#
#     print(refs)
#     return mklab_Garbage
# clear_ram()
# # mpl.rcParams['figure.dpi'] = 300
# # print("\tsetting mpl.rcParams['figure.dpi'] = 300")
def show_system_fonts():
    print(mpl.matplotlib_fname())  # here's the rc file saved. Go mess with it!
    print(plt.rcParams)  # prints the rc file

    def make_html(fontname):
        return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(
            font=fontname
        )

    code = "\n".join(
        [
            make_html(font)
            for font in sorted(
                set([f.name for f in mpl.font_manager.fontManager.ttflist])
            )
        ]
    )

    display(HTML("<div style='column-count: 2;'>{}</div>".format(code)))


def show_palettes():
    cmaps = {}
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(
            top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99
        )
        axs[0].set_title(f"{category} colormaps", fontsize=14)

        for ax, name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect="auto", cmap=plt.get_cmap(name))
            ax.text(
                -0.01,
                0.5,
                name,
                va="center",
                ha="right",
                fontsize=10,
                transform=ax.transAxes,
            )

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()

        # Save colormap list for later.
        cmaps[category] = cmap_list

    plot_color_gradients(
        "Qualitative",
        [
            "Pastel1",
            "Pastel2",
            "Paired",
            "Accent",
            "Dark2",
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
        ],
    )
    plot_color_gradients(
        "Diverging",
        [
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ],
    )
    plot_color_gradients(
        "Perceptually Uniform Sequential",
        ["viridis", "plasma", "inferno", "magma", "cividis"],
    )


def hex_to_RGBA(value):
    value = value.lstrip("#")
    lv = len(value)
    rgba = tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
    rgbaN = (e / 255 for e in rgba)
    return rgbaN

def make_cmap_saturation(
    undersat: tuple = (0.5, 0.0, 0.0),
    oversat: tuple = (0.0, 0.7, 0.0),
    n: int = 100,
):
    """Make a colormap that displays max and lowest (over and undersaturation) of values

    :param undersat: RGB tuple of undersaturated color
    :type undersat: tuple
    :param oversat: RGB tuple of oversaturated color
    :type oversat: tuple
    :param n: number of colors to generate, defaults to 50
    :type n: int, optional
    :return: matplotlib colormap
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    from colour import Color
    from matplotlib.colors import ListedColormap

    ### Create a custom colormap from scratch
    # * Create a list of colors
    colors = list(Color("black").range_to(Color("white"), n))
    colors = [c.rgb for c in colors]  # * Convert to RGB

    ### Add a color for values under and over the range of the colormap
    colors.append(oversat)
    colors.insert(0, undersat)
    custom_cmap = ListedColormap(colors, N=len(colors))

    return custom_cmap


# @timeit  ## Trying to fix matplotlib memory leak
def delete_fig(fig, gc_collect=True, iterations=2):
    """Matplotlib has memory leak issues. This might solve some of it. Maybe."""

    old_backend = mpl.rcParams["backend"]
    mpl.use("Agg")

    plt.savefig(f"temp.png")
    os.remove(f"temp.png")

    # plt.ioff()

    for i in range(iterations):
        fig.clf()
        plt.close(fig)  # clears figure

    del fig
    if gc_collect:  ## can take some time and be inefficient
        gc.collect()

    # plt.ion()
    # mpl.use('module://matplotlib_inline.backend_inline')
    mpl.use(old_backend)


# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


""" 8 Seaborn >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' 8 Seaborn  ">
# °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# %config InlineBackend.figure_format='retina'
# import seaborn as sns
#
# # def ax_dict(g):
# #     '''create a dictionary that maps name of  facetgrid rows and columns to subplots
# #     dic[(row,col)] = ax '''
# #     axes = g.axes.flat
# #
# #     dic = {}
# #     for ax in axes:
# #         axTit = ax.get_title()
# #         if "|" in axTit:
# #             col = axTit.split(" | ")[0].split(" = ")[1]
# #             row = axTit.split(" | ")[1].split(" = ")[1]
# #             dic[(row,col)] = ax
# #         else:
# #             f = axTit.split(" = ")[1]
# #             dic[f] = ax
# #     return dic
#
# def paired_dots(g, connect="all", indices=None):
#     '''
#     Connects each datapoint with its respective paired datapoint
#     :param g:
#     :param connect:
#     :return:
#     '''
#     axes = g.axes.flat
#     for ax in axes:
#         ### Initialize all lines
#         dotsPerAx = len(ax.collections)
#         indices = [(i, i+1) for i in range(0, dotsPerAx-1, 1)]
#
#         ### DRAW ALL LINES
#         for i in indices:
#             ### Retrieve x and y coordinates of each individual point
#             p1 = ax.collections[i[0]].get_offsets()
#             p2 = ax.collections[i[1]].get_offsets()
#             for (x0, y0), (x1, y1), in zip(p1, p2):
#                 ax.plot([x0, x1], [y0, y1], color='black', ls=':', zorder=2)
#
#         ### DRAWW CHOSEN LINES
#         if connect != "all":
#             ### Remove every k-th element
#             k = connect
#             del indices[k-1::k]
#         #print(connect)
#         #print(indices)
#
#         for i in indices:
#             ### Retrieve x and y coordinates of each individual point
#             p1 = ax.collections[i[0]].get_offsets()
#             p2 = ax.collections[i[1]].get_offsets()
#             for (x0, y0), (x1, y1), in zip(p1, p2):
#                 ax.plot([x0, x1], [y0, y1], color='black', ls='-', zorder=2)

# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^


""" 9 Statistics >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
# <editor-fold desc="''' 9 Statistics ">


# from scipy.stats import gaussian_kde, norm
# from scipy.signal import find_peaks
# from scipy.stats import
# from statannot import add_stat_annotation
# from lmfit.models import GaussianModel, RectangleModel, ConstantModel, SkewedGaussianModel

# def splitSphericity(data, by, dv:str, within, subject, report=False):
#     '''
#     splitSphericity(data=DF, by=By, dv=Y, within="Groups")
#     Calculate multiple sphericity tests after grouping the dataframe by other factors.
#     splitSphericity(data=DF, by=By, dv=Y, within="Groups")
#     '''
#
#     print(f"  Dep. Variable:\t {dv}")
#     groups = data.groupby(by)
#     DFdict = {}
#     for level, df in groups:
#         if report== True:
#             pretty_print(df)
#         ### perform test
#         sphr = pg.sphericity(data=df, dv=dv, within=within, subject=subject)
#         print(sphr)
#         sphr = sphr._asdict()
#         ### Display groups so we understand what we did
#         g = [g for g in df.groupby(within)]
#         sphr["groups"] = str([e[0] for e in g])
#         sphr["n per groups"] = str([len(e[1]) for e in g])
#         ### concat
#         DFdict[level] = pd.DataFrame(data=sphr, index=[0])
#
#     spherDF = pd.concat(DFdict.values(), keys=DFdict.keys())
#     pretty_print(spherDF)
#
# def stars(fl:float, a=0.05):
#     if   fl> a:
#         stars = "ns"
#     elif fl<=a/1   and fl>a/5:
#         stars = "*" # use other stars ☆  ★ ★ ★
#     elif fl<=a/5  and fl>a/50:
#         stars = "★ ★"
#     elif fl<=a/50 and fl>a/500:
#         stars = "* * *"
#     elif fl<=a/500:
#         stars = "* * * *"
#     else:
#         stars = float("NaN")
#     return stars
# def effectSize(fl:float, t=(.01, .06, .14, .5)):
#     if   fl<t[0]:
#         effectSize = "No Effect"
#     elif fl>=t[0] and fl<t[1]:
#         effectSize = "Small"
#     elif fl>=t[1] and fl<t[2]:
#         effectSize = "Medium"
#     elif fl>=t[2] and fl<t[3]:
#         effectSize = "Large"
#     elif fl>=t[3]:
#         effectSize = "Huge"
#     else:
#         effectSize = float("NaN")
#     return effectSize
#
# def pairedTTests(data, dv:str, within:list, subject:str,
#                  padjust="fdr_bh", alpha=0.05,
#                  switchFactors=True):
#
#     ### Check if multiple factors are mentioned in within
#     has_multifactors = type(within)==list and len(within) > 1
#     if type(within)==list and len(within) ==1:
#         within = within[0]
#
#     ### When within has two elements, it DOES matter which order you post them
#     if has_multifactors:
#         w1 = within
#         if switchFactors:
#             w2 = [within[1], within[0]]
#         else:
#             w2=w1
#         print(f"### PairedTTests: Multiple Factors detected:\nwithin1: \n{w1}within2: {w2}")
#     else:
#         print(f"### PairedTTests: One Factor detected: {within}")
#         w1 = within
#         w2 = within # just do it twice, the merging will remove duplicate rows
#
#     ### Perform Tests
#     ph1 = pg.pairwise_ttests(data=data, dv=dv, within=w1, subject=subject, padjust=padjust, nan_policy="pairwise")
#     ph2 = pg.pairwise_ttests(data=data, dv=dv, within=w2, subject=subject, padjust=padjust, nan_policy="pairwise")
#     ### Merge dataframes (don't concat, or you'll get duplicate rows)
#     PH = ph1.merge(ph2, how='outer')
#     ### The column of the second factor concatenated at the end, place it next to the first factor
#     if switchFactors==True and has_multifactors:
#         PH.insert(2, within[1], PH.pop(within[1]))
#
#     ### Annotate Stars
#     PH["**-unc"] = PH["p-unc"].apply(stars, alpha)
#     if padjust!="none":
#         PH["**-corr"] = PH["p-corr"].apply(stars, alpha)
#
#     PHsig = PH[PH["p-unc"]<alpha] # Take only the rows with significant p-values
#
#     return PH, PHsig
#
# def splitPairedTTests(data, dv:str, by, within:list, subject:str,
#                       padjust="fdr_bh", alpha=0.05,
#                       switchFactors=True):
#     ''' splits a dataframe by one Factor into dataframes of that factors levels and performs multiple paired ttests per level
#     dv = str dependent variable
#     by = str or list that indicates the multiindex levels by which the graphic was splitted by
#     '''
#     ### check if multiple factors are mentioned in within
#     has_multifactors = type(within)==list and len(within) > 1
#     if type(within)==list and len(within) ==1:
#         within = within[0]
#
#     ### When within has two elements, it DOES matter which order you post them
#     DFdict={}
#     if has_multifactors:
#         w1 = within
#         if switchFactors:
#             w2 = [within[1], within[0]]
#         else:
#             w2=w1
#         print(f"### splitPairedTTests: Multiple Factors detected:\nwithin1: \n{w1}within2: {w2}")
#     else:
#         print(f"### splitPairedTTests: One Factor detected: {within}")
#         w1 = within
#         w2 = within # just do it twice, the merging will remove duplicate rows
#
#     for name, df in data.groupby(by):
#         #pretty_print(df)
#         #print("w1", w1)
#         #print("w2", w2)
#         #print("subject", subject)
#         ph1 = pg.pairwise_ttests(data=df, dv=dv, within=w1, subject=subject, padjust=padjust, nan_policy="pairwise")
#         ph2 = pg.pairwise_ttests(data=df, dv=dv, within=w2, subject=subject, padjust=padjust, nan_policy="pairwise")
#         ### Merge dataframes (don't concat, or you'll get duplicate rows)
#         ph = ph1.merge(ph2, how='outer')
#         ### The column of the second factor concatenated at the end, place it next to the first factor
#         if switchFactors==True and has_multifactors:
#             ph.insert(2, within[1], ph.pop(within[1]))
#
#         ### Annotate Stars
#         ph["**-unc"] = ph["p-unc"].apply(stars, alpha)
#         if padjust!="none":
#             ph["**-corr"] = ph["p-corr"].apply(stars, alpha)
#
#         DFdict[name] = ph
#
#     ### Summarize Results
#     byL = [by]  if type(by)==str else by # make list if it contains only one value
#     PH = pd.concat(DFdict.values(), keys=DFdict.keys(), names=byL)
#     PHSig = PH[PH["p-unc"]<alpha]# Take only the rows with significant p-values
#
#     return PH, PHSig
#
# def constructPairs(ph, factors):
#     '''takes the output from pg.pairwise_ttests(), and constructs a list of pairs and their p-values
#     that can be taken up by statannot
#     ph = dataframe with post hoc analysis from pg.pairwise_ttests
#     '''
#     ### Check if factors is a single string or a list
#
#     has_multifactors = type(factors)==list and len(factors) > 1
#     if type(factors)==list and len(factors) ==1:
#         factors = factors[0]
#         #print(factors)
#
#     pairs = []
#     pvals = []
#     outDict = {}  # {pair: pval}
#     if not has_multifactors:
#         print("ONE FACTOR DETECTED")
#         for _, row in ph.iterrows():
#             pair = (row["A"], row["B"]) # no need to include factors, since only one factor, statannot detects that
#             pairs.append(pair)
#             pvals.append(row["p-unc"] )
#             outDict[pair] = row["p-unc"]
#     else:
#         print("TWO FACTORS DETECTED")
#         ### Take only the rows with each factor specified
#         phInteract = ph[ph["Contrast"].str.contains('\*')]
#         for _, row in phInteract.iterrows():
#             ## Fcol can be either from post-hoc1 or post-hoc2 depending on the order of the passed factors
#             factor = row[factors[0]]
#             if pd.notna(factor): # switch column if NaN, also check: if not math.isnan(factor)
#                 pair = ((factor, row["B"]),
#                         (factor, row["A"]) )
#             else:
#                 factor = row[factors[1]]
#                 pair = ((row["B"], factor),
#                         (row["A"], factor))
#
#             ## Gather Results
#             pairs.append(pair)
#             pvals.append(row["p-unc"] )
#             outDict[pair] = row["p-unc"]
#
#     #print(outDict)
#     return pairs, pvals, outDict
#
#
# def splitRmANOVA(data, dv:str, by:str, within:list, subject:str, effsize="ng2", alpha=0.05):
#     """ Splits a dataframe which has one Factor into levels and performs repeated measures ANOVA per these levels
#     ! Three-way ANOVA not supported
#     """
#
#     factorial = "Two-Factor"   if len(within)==2  else ""
#     print(f"{factorial} Repeated Measures ANOVA, alpha = {alpha}, effectsize = {effsize}")
#     print("  dVariable:  \t", dv)
#     print("  Split by:   \t", by)
#     print("  Within:     \t", within)
#     print("  Subject:    \t", subject)
#
#     gr = data.groupby(by)
#     for name, df in gr:
#         print(f"\n >> Testing '{name}' from {by}:")
#
#         aov = pg.rm_anova(data=df, dv=dv, within=within, subject=subject,
#                           effsize=effsize, detailed=True)
#
#         aov["**-unc"] = aov["p-unc"].apply(stars, alpha)
#         #aov["**-GG"] = aov["p-GG-corr"].apply(stars, alpha)
#         aov["EffSize"] = aov[effsize].apply(effectSize)
#         pretty_print(aov)
#
# def splitRmANOVA2(data, dv:str, by:str, within:list, subject:str, alpha=0.05):
#     ''' splits a dataframe which has one Factor into its levels and performs repeated measures ANOVA per level '''
#
#     gr = data.groupby(by)
#     for name, df in gr:
#         print(f"\n{name}:")
#         aov = anova.AnovaRM(data=df, depvar=dv, within=within, subject=subject).fit()
#         print(aov)

# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^
