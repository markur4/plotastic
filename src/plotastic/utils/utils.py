"""Small utility functions that are frequently used in multiple modules across complete
library 
"""

# %% Imports: Just the re-used ones, special ones are imported in the functions


from decimal import Decimal
import collections

from pathlib import Path

import warnings

import re

from IPython.display import DisplayObject  #' For type hinting of ut.pp()
from IPython import get_ipython


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import plotastic.caches as caches

from typing import TYPE_CHECKING, Hashable

if TYPE_CHECKING:
    pass


# %%
# == Types =============================================================

NUMERICAL_TYPES = [
    int,
    float,
    complex,
    Decimal,
    np.number,
]


# %%
# == print =============================================================


def get_terminal_width():
    import shutil

    try:
        # Get the terminal width
        columns, _ = shutil.get_terminal_size(fallback=(80, 24))
        return columns
    except Exception:
        # Handle exceptions if the terminal size cannot be determined
        return 80  # Default value


def print_separator(char="=", length=None):
    # Get the terminal width
    if length is None:
        length = get_terminal_width()

    # Calculate the number of characters required
    num_chars = length // len(char)

    # Print the separator line
    print(char * num_chars)


# %%
# == Builtins: ALL =====================================================


def get_type(o):
    """gets the type of an object as a string, e.g. 'int' or 'str'"""
    return str(type(o)).split("'")[1]


def catch_duplicates(l: list | tuple) -> list:
    """Checks if all elements of a list (x) are within another list (y)"""
    l = (l,) if isinstance(l, str) else l
    return [item for item, count in collections.Counter(l).items() if count > 1]


def check_unordered_identity(
    o1: str | tuple | list, o2: str | tuple | list, ignore_duplicates=False
) -> bool:
    """both objects should be the same type. But we don't care about order of elements"""
    if not isinstance(o1, str) and not isinstance(o2, str):
        if set(o1) == set(o2):
            if len(o1) != len(o2) and not ignore_duplicates:
                dupl_o1 = catch_duplicates(o1)
                dupl_o2 = catch_duplicates(o2)
                raise AssertionError(
                    f"Both lists have matching elements, but at least one has duplicate elements: {dupl_o1}, {dupl_o2} "
                )
            return True
    elif isinstance(o1, int) and isinstance(o2, int):
        return o1 == o2
    elif isinstance(o1, str) and isinstance(o2, str):
        return o1 == o2
    else:
        raise AssertionError(
            f"#! {o1} is not comparable to {o2} (Did you miss the comma in your tuple?)"
        )


# %%
# == Builtins: Numbers =================================================


def exponent_from_float(number: float):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def mantissa_from_float(number: float):
    return Decimal(number).scaleb(-exponent_from_float(number)).normalize()


# %%
# == Builtins: Strings =================================================


def capitalize(s: str) -> str:
    """Takes first word of string and capitalizes it, e.g. 'conc.: 1 mL'-> 'Conc.: 1 mL'"""
    # s = "conc.: 1 mL"
    s1 = s.split(" ")[0].capitalize()
    return s1 + " " + " ".join(s.split(" ")[1:])


def re_matchgroups(pattern, string: str, flags=None) -> list[dict]:
    """Takes a regular expression searchpattern that includes group
    names (?P<name>...) and returns a list of dictionaries with
    groupnames as keys and matched strings as values

    :param pattern: compiled re searchpattern, e.g. from re.compile(".*")
    :param string: str,
    :param flags: e.g. re.MULTILINE or re.DOTALL :returns dict

    :example:

    >>> string = "abc abc2 abc3"
    >>> pattern = re.compile(r'(?P<WORD>abc)(?P<INDEX>\d)')
    >>> matches = re_matchgroups(pattern=pattern, string=string)
    >>> matches
    [{'WORD': 'abc', 'INDEX': '2'},
    {'WORD': 'abc', 'INDEX': '3'}]
    """

    return [
        match.groupdict()
        for match in re.finditer(pattern=pattern, string=string, flags=flags)
    ]


# %%


def print_indented(s: str, indent: str = "\t"):
    """Print a string `s` indented with `n` tabs at each newline"""
    for x in s.split("\n"):
        print(indent + x)


def string_to_words(s: str) -> list[str]:
    """Splits a string into words, removing all newlines and tabs
    e.g. 'conc.: 1 mL'-> ['conc.:', '1', 'mL']"""

    ### Remove any newline and tabs
    s = s.replace("\t", " ").replace("\n", " ")

    ### Split into words
    words = s.split(" ")
    # from icecream import ic
    # ic(words)

    ### Remove empty words and strip whitespace
    words = [w.strip() for w in words if len(w.strip()) > 0]

    return words


if __name__ == "__main__":
    descr = """These are the 5 groups with the largest
                    samplesizes:"""
    w = string_to_words(descr)
    print(w)

# %%


def wrap_text(
    string: str,
    width: int = 72,
    width_first_line: int = None,
    indent: str = None,
) -> str:
    """Wraps a multiline string into a certain width. If first_line is
    specified, it will remove those characters from the first line
    before wrapping.

    :param string: A multiline string
    :type string: str
    :param width: Width of characters to wrap to, defaults to 72
    :type width: int, optional
    :param width_first_line: Width of characters first line, defaults to
        None because of that's how much space declaring the :param param:
        takes
    :type width_first_line: int, optional
    :param indent: Indentation to add to each line, defaults to None
    :type indent: str, optional
    :return: Wrapped string
    :rtype: str
    """
    ### Remove all previous formatting, newlines, tabs, spaces etc.
    words = string_to_words(string)

    ### Return if string is already short enough
    if len(" ".join(words)) <= width:
        TEXT = " ".join(words)
        return TEXT

    # == Wrap text ==#
    TEXT = ""

    ### Wrap first line.
    # # Not indented
    if not width_first_line is None:
        while len(TEXT + words[0]) < width_first_line:
            TEXT += words.pop(0) + " "
            if not words:
                break  # # If there are words left
        TEXT = TEXT[:-1]  # # Remove last space
        TEXT += "\n"

    ### Wrap remaining lines
    lines = []
    init_line = lambda: indent if not indent is None else ""
    line = init_line()
    while words:
        if len(line + words[0]) <= width:
            line += words.pop(0) + " "
            if not words:
                break  # # If there are words left
        else:
            lines.append(line.rstrip())
            line = init_line()
    if line:
        lines.append(line.rstrip())

    ### Join lines, add e.g. 12 spaces as indent
    TEXT += "\n".join(lines)

    return TEXT


if __name__ == "__main__":
    descr = """
    Mode of overwrite protection. If "day", it simply adds the current
    date at the end of the filename, causing every output on the same
    day to overwrite itself. If "nothing" ["day", "nothing"], files with
    the same filename will be detected in the current work directory and
    a number will be added to the filename. If True, everything will be
    overwritten.
    """
    w = wrap_text(descr)
    print(w)


# %%
# == Builtins: Lists ===================================================


def ensure_list(
    s: str | list | tuple | str | Hashable | None,
    allow_none=True,
    convert_none=False,
) -> list | None:
    """Converts Element into a list, even if it's just one"""
    if s is None:
        if convert_none:
            return [
                None,
            ]
        elif allow_none:
            return None
        else:
            raise TypeError(
                f"#! Must pass tuple, None not allowed. ({s} was passed)"
            )
    elif isinstance(s, list):
        return s
    elif isinstance(s, (tuple, set)):
        return list(s)
    else:
        return [
            s,
        ]


def index_of_matchelements(i1: list, i2: list):
    """If i1 is ['F1', 'F2', 'F3'] and i2 is['F1', 'F3'], return [0, 2]"""
    return [i1.index(i) for i in i2]


def flatten(s: list | tuple, np=False) -> list:
    """

    :param s: Listor tuple or iterable
    :param np: If list is deeper than one Nesting (e.g. 3D), use numpy flatten method.
    Otherwise it'll perform list comprehension
    :return:
    """
    if np:
        return np.array(s, dtype="object").flatten(order="K").tolist()
    else:
        return [e for sublist in s for e in sublist]


def get_duplicate(s: list | tuple):
    result = [
        item for item, count in collections.Counter(s).items() if count > 1
    ]
    if len(result):
        return result[0]
    else:
        return result


def all_of_l1_in_l2(l1: list | tuple, l2: tuple | list) -> bool:
    """Checks if all elements of a list (x) are within another list (y)"""
    l1 = (l1,) if isinstance(l1, str) else l1
    l2 = (l2,) if isinstance(l2, str) else l2
    return all(tuple(e in l2 for e in l1))


# %%
# == Builtins: Dicts ===================================================


def get_key(q, dic):
    """

    :param val:
    :param dic: {k1: [1,2,3,"et"], k2: [4,"wsef"]}
    :return: key if q is in val
    """
    result = "not found"
    for k, v in dic.items():
        if q in v:
            result = k

    return result


def update_dict_recursive(d: dict, u: dict) -> dict:
    """Recursively update a dict.

    Args:
        d (dict): Dictionary to be updated
        u (dict): Dictionary to update with

    Returns:
        dict: New dictionary
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def remove_None_recursive(d: dict) -> dict:
    """Recursively remove None values from a dict.

    Args:
        d (dict): Dictionary to be updated

    Returns:
        dict: New dictionary
    """

    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, collections.abc.Mapping):
            d[k] = remove_None_recursive(v)
        elif v is None:
            del d[k]
    return d


def printable_dict(D, key_adjust=5, start_message=None, print_type=True):
    s = "\n######" + start_message if start_message else "\n######"
    for k, v in D.items():
        key = f"'{k}'" if type(k) is str else k
        val = f"'{v}'" if type(v) is str else v
        s += f"\n  |# {key.ljust(key_adjust)}: {val}"
        if print_type:
            s += f"\t {type(v)}"
        s += f"\t#|"
    s += "\n"
    return s


# %%

# %%
# == pandas ============================================================


def pp(df, prec: int = None, ret: bool = False) -> DisplayObject:
    """
    Displays pandas dataframe to jupyter notebook
    :param df:
    :param prec: display precision of floats
    :param ret:  If True (default), also returns the dataframe
    :return:
    """
    from IPython.display import display, HTML

    if prec:
        # with pd.option_context('display.float_format', '${:,.2f}'.format):
        with pd.option_context("display.precision", prec):
            display(HTML(df.to_html().replace("\\n", "<br>")))
    else:
        display(HTML(df.to_html().replace("\\n", "<br>")))

    if ret:
        return df


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


def drop_columns_by_regex(DF: "pd.DataFrame", pattern: str):
    DF = DF[DF.columns.drop(list(DF.filter(regex=pattern)))]
    return DF


# %%
#
# == Plotting ==========================================================


def get_bbox_width(bbox: mpl.transforms.Bbox, in_inches=True):
    if not isinstance(bbox, mpl.transforms.Bbox):
        try:
            # bbox = bbox.get_window_extent() # ?? same as get_tightbbox()
            bbox = bbox.get_tightbbox()
        except AttributeError:
            raise TypeError(
                f"#! bbox must be of type matplotlib.transforms.Bbox, not {type(bbox)}"
            )

    ### BBox is [[xmin, ymin], [xmax, ymax]]
    xmin, xmax = bbox.extents[0], bbox.extents[2]
    # ymin, ymax = bbox.extents[1], bbox.extents[3]

    width_pixels = xmax - xmin
    if in_inches:
        width_inches = width_pixels / plt.rcParams["figure.dpi"]
        return width_inches
    else:
        return width_pixels


def get_bbox_height(bbox: mpl.transforms.Bbox, in_inches=True):
    if not isinstance(bbox, mpl.transforms.Bbox):
        try:
            # bbox = bbox.get_window_extent() # ?? same as get_tightbbox()
            bbox = bbox.get_tightbbox()
        except AttributeError:
            raise TypeError(
                f"#! bbox must be of type matplotlib.transforms.Bbox, not {type(bbox)}"
            )

    ### BBox is [[xmin, ymin], [xmax, ymax]]
    # xmin, xmax = bbox.extents[0], bbox.extents[2]
    ymin, ymax = bbox.extents[1], bbox.extents[3]

    height_pixels = ymax - ymin
    if in_inches:
        height_inches = height_pixels / plt.rcParams["figure.dpi"]
        return height_inches
    else:
        return height_pixels


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
    #' Create a list of colors
    colors = list(Color("black").range_to(Color("white"), n))
    colors = [c.rgb for c in colors]  #' Convert to RGB

    ### Add a color for values under and over the range of the colormap
    colors.append(oversat)
    colors.insert(0, undersat)
    custom_cmap = ListedColormap(colors, N=len(colors))

    return custom_cmap


# !! Cache it
make_cmap_saturation = caches.MEMORY_UTILS.subcache(make_cmap_saturation)

# %%
# == MPL: Fonts ====================================================


def mpl_font():
    return mpl.font_manager.FontProperties().get_name()


def mpl_fontpath():
    """Returns path to the font used by matplotlib"""
    from matplotlib.font_manager import findfont, FontProperties

    family = mpl.rcParams["font.family"]
    properties = FontProperties(family=family)
    font = findfont(properties)
    return font


def mpl_fontsizes_get_all() -> dict:
    ### Make dummy text object
    _fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Text")

    ### These are scaled dependend on mpl.rcParams["font.size"]!
    fonts = [
        "xx-small",
        "x-small",
        "small",  #' if medium is 10, then this is 8.33
        "medium",  #' = mpl.rcParams["font.size"]
        "large",  #' if medium is 10, then this is 12.0
        "x-large",
        "xx-large",
        "larger",
        "smaller",
    ]

    fontsizes = {}
    for font in fonts:
        t.set_fontsize(font)
        fontsizes[font] = round(t.get_fontsize(), 2)

    plt.close()  #!!
    return fontsizes


def mpl_fontsize_from_rc(rc_param: str = "font.size") -> int:
    """Returns fontsize from rcParams."""
    fontsize = mpl.rcParams[rc_param]
    if isinstance(fontsize, str):
        return mpl_fontsizes_get_all()[fontsize]
    elif isinstance(fontsize, (int, float)):
        return fontsize


# %%

# == I/O ===============================================================


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


def copy_by_pickling(obj, plt_close=True):
    """converts to bytes and loads it to return it"""
    import io
    import pickle
    from matplotlib import pyplot as plt

    #' CONVERT TO BYTE RAM
    # with io.BytesIO() as buf: # !! 'with' statement not working with pyplot
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)

    #' RELOAD IT
    copy = pickle.load(buf)
    # p2 = pickle.loads(buf.getvalue()) # THIS GETS VALUE WITHOUT RESETTING buf

    #' MPL IS ANNOYING
    # if plt_close:
    plt.close()

    return copy


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


# %% Warnings and exceptions


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
