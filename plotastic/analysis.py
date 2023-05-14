from __future__ import annotations  # for type hinting my Class type for return values
import numpy as np
import pandas as pd
from copy import copy
from typing import Dict, List, Callable, TYPE_CHECKING

# from dataclasses import dataclass
# import importlib

import warnings

# from collections import OrderedDict
import seaborn as sns
from scipy.stats import skew as skewness
import matplotlib.pyplot as plt

# import markurutils.UTILS as ut
# from markurutils.builtin_types import printable_dict
# from markurutils.filer import Filer
import markurutils as ut



df = None  # * Prevent warning when using catchstate


def catchstate(df, var_name: str = "df"):
    ## THIS FUNCTIONS DOESN'T WORK WHEN IMPORTED FROM MODULE
    """
    Helper function that captures intermediate Dataframes.
    In the global namespace, make a new variable called var_name and set it to dataframe
    :param df: Pandas dataframe
    :param var_name:
    :return:
    """
    globals()[var_name] = df

    return df


### Dims #.......................................................................................................


# @dataclass
class Dims:
    def __init__(
        self,
        y: str = None,
        x: str = None,
        hue: str = None,
        row: str = None,
        col: str = None,
    ):
        self.y = y
        self.x = x
        self.hue = hue
        self.row = row
        self.col = col
        self._by = None
        # self._by_str = None

    @property
    def by(self) -> list[str] | None:
        if self._by:
            return self._by
        elif self.row and self.col:
            return [self.row, self.col]
        elif self.row:
            return [self.row]
        elif self.col:
            return [self.col]
        else:
            return None

    def asdict(self, incl_None=True, incl_by=True) -> dict:
        d = dict(y=self.y, x=self.x, hue=self.hue, row=self.row, col=self.col)
        if incl_by:
            d.update(dict(by=self.by))
        if not incl_None:
            d = {k: v for (k, v) in d.items() if (not v is None)}
        return d

    def set(self, inplace=False, **kwargs) -> "Dims":
        newobj = self if inplace else copy(self)
        for k, v in kwargs.items():
            v = v if not v == "none" else None
            setattr(newobj, k, v)
        return newobj

    def getvalues(self, keys: list[str] | tuple[str], *args):
        """
        Converts a list of dimensions into a list of dimension values, e.g.
        :param keys: ["x", "y", "col"]
        :return: e.g. ["smoker", "tips", "day"]
        """
        defkeys = ("x", "y", "hue", "row", "col")
        l = []
        keys = [keys] + [arg for arg in args]
        for key in keys:
            assert key in defkeys, f"#! '{key}' should have been one of {defkeys}"
            l.append(getattr(self, key))
        return l

    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> "Dims":
        """
        Set attributes. Detects Duplicates, switches automatically
        :param keys: Two dimensions to switch. Only 2 Positional arguments allowed. Use e.g. dims.switch("x", "hue", **kwargs)
        :param inplace: Decide if this switching should change the dims object permanently (analogously to pandas dataframe). If False, you should pass return value into a variable
        :param verbose: Whether to print out switched values
        :param kwarg: e.g. dict(row="smoker")
        :return: dims object with switched parameters
        """

        """HANDLE ARGUMENTS if keys are passed, e.g. dims.switch("x","row",**kwargs)"""
        if len(keys) == 0:
            pass
        elif len(keys) == 2:
            assert (
                len(kwarg) == 0
            ), f"#! Can't switch when both keys and kwarg is passed"
            values = self.getvalues(*keys)
            kwarg[keys[0]] = values[1]
        else:
            raise AssertionError(f"#! '{keys}' should have been of length 2")
        assert len(kwarg) == 1, f"#! {kwarg} should be of length 1 "

        """PRINT FIRST LINE"""
        if verbose:
            todo = "RE-WRITING" if inplace else "TEMPORARY CHANGING:"
            print(
                f"#! {todo} {self.__class__.__name__} with keys: '{keys}' and kwarg: {kwarg}:"
            )
            print("   (dim =\t'old' -> 'new')")

        ### SWITCH IT
        ### COPY OBJECT
        oldby = self.by
        original: dict = copy(
            self.asdict(incl_None=True),
        )
        newobj = self if inplace else copy(self)

        qK, qV = *kwarg.keys(), *kwarg.values()
        replace_v = "none"
        for oK, oV in original.items():  # Original Object
            if qK == oK:
                replace_v = oV
                setattr(newobj, qK, qV)
            elif qK != oK and oV == qV:
                replace_v = original[qK]
                setattr(newobj, oK, replace_v)
        assert (
            replace_v != "none"
        ), f"#! Did not find {list(kwarg.keys())} in dims {list(original.keys())}"

        ### PRINT THE OVERVIEW OF THE NEW MAPPING"""
        if verbose:
            for (oK, oV), nV in zip(original.items(), newobj.asdict().values()):
                pre = "  "
                if oV != nV and oV == replace_v:  # or replace_v == "none":
                    printval = f"'{replace_v}' -> '{qV}'"
                    pre = ">>"
                elif oK == "by" and newobj.by != oldby:
                    printval = (
                        f"'{oldby}' -> '{newobj.by}'"
                        if type(newobj.by) is str
                        else f"{oldby} -> {newobj.by}"
                    )
                elif oV != nV and oV != replace_v:
                    printval = f"'{oV}' -> '{replace_v}'"
                    pre = " <"
                else:  # oV == nV
                    printval = f"'{oV}'" if type(oV) is str else f"{oV}"
                if len(oK) < 3:
                    oK = oK + "  "

                printval = printval.replace("'None'", "None")  # REMOVE QUOTES

                print(f" {pre} {oK} =\t{printval}")

        ### x AND y MUST NOT BE None"""
        assert not None in [self.y, self.x], "#! This switch causes x or y to be None"

        return newobj


### ANALYSIS ===================================================


class Analysis:
    def __str__(self):
        # d = self.__dict__
        D = {
            a: getattr(self, a)
            for a in dir(self)
            if (
                not a.startswith("_")
                and not callable(getattr(self, a))
                and not isinstance(getattr(self, a), ut.Filer)
            )
        }

        ### Catch unprintable types
        if type(D.get("data")) is pd.DataFrame:
            D["data"] = (D["data"].shape, list(D["data"].columns))
        if "fig2nd" in D:
            D["fig2nd"] = f"{len(D['fig2nd'])} axes"
        # if "ax" in D:
        #     D["ax"] = len(self.ax)

        return ut.printable_dict(D=D, start_message=f"{type(self)}: ")

    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict | Dims,
        title="untitled",
        # transform=None
    ):
        self.data = data
        self.dims = dims if type(dims) is Dims else Dims(**dims)
        self._title = title  # needed as property so that setter updates filer

        self.is_transformed = False
        self.transform_funcs = []  ### HISTORY OF TRANSFORMATIONS
        self._y_untransformed = (
            self.dims.y
        )  ### STORE IT SO WE CAN RESET IT AFTER TRANSFORMATION
        # self._dv = self.dims.y
        # self.transform_func = transform
        # self.dv_untransformed = self.dims.y
        # if transform:
        #     self.add_transform_col()

        self._factors_all = None  # [x, hue, row, col] defined in dims
        self._factors_xhue = None
        self._factors_rowcol = None

        self._levels = None
        """WARN USER IF SOME FACETS ARE EMPTY """
        self.get_empty_groups()
        self._vartypes = (
            None  # Categorical or Continuous? (Nominal? Ordinal? Discrete? Contiuous?)
        )

        self._tree = None

        """# Use the pyrectories Filer object if pyrectories is installed"""
        # if importlib.util.find_spec("pyrectories"):
        #    from pyrectories import Filer
        # else:
        # from markurutils.filer import Filer

        self.filer = ut.Filer(title=title)

    ### TITLE .......................................................................................................'''

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.filer.title = value

    def add_to_title(
        self, to_end: str = "", to_start: str = "", con: str = "_", inplace=False
    ) -> "Analysis":
        """
        :param to_start: str, optional (default="")
        String to add to start of title
        :param to_end: str, optional (default="")
        String to add to end of title
        :param con: str, optional (default="_")
        Conjunction-character to put between string addition and original title
        :return: str
        """
        a = self if inplace else ut.copy_by_pickling(self)

        if to_start:
            a.title = f"{to_start}{con}{a.title}"
        if to_end:
            a.title = f"{a.title}{con}{to_end}"
        return a

    ### Factors .....................................................................................................'''

    def getfactors(self, putative_factors: str | list[str,]) -> str | list[str]:
        """Get column name, if "x" or "hue" is passed instead of actual column name"""
        if isinstance(putative_factors, str):
            putative_factors = [putative_factors]

        result = []
        dimsD = self.dims.asdict()
        for f in putative_factors:
            if f in self.data:
                result.append(f)
            elif f in dimsD:
                factor = dimsD[f]
                # assert not factor is None, f"#! Can't set {putative_factors} as factors, since '{f}' is assigned to None.\n\tUse one of {self.dims.asdict(incl_None=False, incl_by=False)} "
                result.append(factor)
            else:
                raise AssertionError(
                    f"#!! '{f}' should have been one of these dimensions or column names: \n\t{list(dimsD.keys() )}\n\t{list(self.data.columns)}"
                )

        if len(result) == 1:
            return result[0]
        else:
            return result

    def get_factor_from_level(self, level: str):
        """Gets the factor from a level"""
        for factor, levels in self.levels.items():
            if level in levels:
                return factor

    def get_rank_from_level(self, level: str):
        """Gets the factor from a level"""
        for rank, levels in self.hierarchy.items():
            if level in levels:
                return rank

    @property
    def factors_all(self) -> list[str]:
        # f = list(np.concatenate((self.factors, self.dims.by)).flat )
        # self._factors_all = list(set(f) )
        f = (self.dims.row, self.dims.col, self.dims.hue, self.dims.x)
        self._factors_all = [e for e in f if (not e is None)]
        return self._factors_all

    @property
    def factors_as_kwargs(self) -> dict:
        """
        gets the dimensions in forms of a dictinary to be passed onto seaborn functions
        :return:
        {"y": self.dims.y, "x": self.dims.x,"hue": self.dims.hue, "col": self.dims.col, "row": self.dims.row}
        :rtype: dict
        """
        return self.dims.asdict(incl_by=False, incl_None=False)
        # return {dim: factor for dim, factor in self.dims.asdict().items() if not factor is None}

    @property
    def factors_xhue(self) -> str | list[str]:
        if self.dims.hue:
            self._factors_xhue = (self.dims.x, self.dims.hue)
        else:
            self._factors_xhue = self.dims.x
        return self._factors_xhue

    @property
    def factors_rowcol(self) -> str | tuple[str] | None:
        if self.dims.row and self.dims.col:
            self._factors_rowcol = (self.dims.row, self.dims.col)
        elif self.dims.row:
            self._factors_rowcol = self.dims.row
        elif self.dims.col:
            self._factors_rowcol = self.dims.col
        else:
            self._factors_rowcol = None
        return self._factors_rowcol

    @property
    def columns_not_factor(self) -> list[str]:
        return [c for c in self.data.columns if c not in self.factors_all]

    ### Levels ......................................................................................................'''

    @property
    def vartypes(self) -> dict:
        D = dict()
        for factor in self._factors_all:
            type = self.data[factor].dtype.name
            if type == "object":
                D[factor] = "object"
                print(
                    f"#! factor '{factor}' is of type object so it's probably a string"
                )
            if type == "category":
                D[factor] = "category"
            elif type in ["int", "float", "float32", "float64", "int32", "int16"]:
                D[factor] = "continuous"
            else:
                print(f"#!!! factor '{factor}' is of unknown type '{type}'")
                D[factor] = "unknown"
        self._vartypes = D
        return self._vartypes

    @property
    def levels(self) -> dict:
        """
        :return: {"f1": [lvl1, lvl2], "f2": [lvl1, lvl2],}
        """
        D = dict()
        for factor in self.factors_all:
            # f = self.getfactors(factor) # ! makes no sense
            S = self.data[factor]
            # if isinstance(S.dtype, pd.api.types.CategoricalDtype):
            if S.dtype.name == "category":
                D[factor] = S.cat.categories.to_list()
            else:
                D[factor] = S.unique()
        self._levels = D
        return self._levels

    @property
    def levels_tuples(self) -> "list[tuple]":
        return [tuple(l) for l in self.levels.values() if not l is None]

    @property
    def levels_xhue_flat(self) -> tuple:
        """
        :return:
        >>> (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
        """
        l = []
        for factor in (self.dims.x, self.dims.hue):
            if factor is None:
                continue
            S = self.data[factor]
            if S.dtype.name == "category":
                [l.append(e) for e in S.cat.categories.to_list()]
            else:
                [l.append(e) for e in S.unique()]
        return tuple(l)

    @property
    def levels_rowcol_flat(self) -> tuple:
        """
        :return: (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
        """

        l = []
        for factor in (self.dims.row, self.dims.col):
            if factor is None:
                continue
            S = self.data[factor]
            if S.dtype.name == "category":
                [l.append(e) for e in S.cat.categories.to_list()]
            else:
                [l.append(e) for e in S.unique()]
        return tuple(l)

    @property
    def levels_hierarchy(self) -> "dict":
        """
        Has ROW (not the factor) as keys!
        :return:
        >>> {"ROW":["r_l1", "row_l2"], "COL":["col_l1", "col_l2"], "HUE":["hue_lvl1", "hue_lvl2"], "X":["..."]}
        """

        D = self.levels
        return {
            "ROW": D.get(self.dims.row),
            "COL": D.get(self.dims.col),
            "HUE": D.get(self.dims.hue),
            "X": D.get(self.dims.x),
        }

    # @property
    # def hierarchy_levels_tuples(self) -> "list[tuple]":
    #     """
    #     :return:
    #     >>> [("row_lvl1", "row_lvl2"), ("col_lvl1", "col_lvl2"), ("hue_lvl1", "hue_lvl2"), ("hue_lvl1", "hue_lvl2")]
    #     """
    #     return [tuple(l) for l in self.hierarchy.values() if not l is None]

    ### DESCRIBE DATA ...............................................................................................'''

    def plot_data(self):
        """Simple plot that shows the data points separated by dimensions"""
        sns.catplot(kind="swarm", data=self.data, **self.factors_as_kwargs)
        plt.show()
        

    def describe_data(self, verbose=False, plot=False):
        
        ### Plot Data
        if plot:
            self.plot_data

        ### Define Functions
        def NaNs(s: "pd.Series"):
            result = int(s.isna().sum())
            if not result is None and result != 0:
                return result
            else:
                return None

        def Zeroes(s: "pd.Series"):
            return s.value_counts().get(0)  ## COLUMNNOT SHOWING IF ALL None

        def Negatives(s: "pd.Series"):
            result = np.sum(np.array(s) < 0)
            if not result is None and result != 0:
                return result
            else:
                return None

        def Q1(s: "pd.Series"):
            return np.percentile(s, [25])[0]

        def Q3(s: "pd.Series"):
            return np.percentile(s, [75])[0]

        def IQR(s: "pd.Series"):
            q3, q1 = np.percentile(s, [75, 25])
            return q3 - q1

        def skew(s: "pd.Series"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skewness(s)

        ### Build DataFrame
        df = (
            pd.pivot_table(
                self.data,
                values=self.dims.y,
                index=self.factors_all,
                aggfunc=[
                    "count",
                    NaNs,
                    Zeroes,
                    Negatives,
                    "median",
                    "mean",
                    "std",
                    Q1,
                    Q3,
                    IQR,
                    skew,
                ],
            )
            .sort_index()
            .convert_dtypes()
        )
        if verbose:
            ut.pp(df)
        return df

    ### Iterate through DATA  .......................................................................................................'''

    def get_empty_groups(self):
        """Detects Facets with empty groups"""

        def NaNs(s: "pd.Series"):
            result = int(s.isna().sum())
            if not result is None and result != 0:
                return result
            else:
                return None

        empties = (
            pd.pivot_table(
                self.data,
                values=self.dims.y,
                index=self.factors_all,
                aggfunc=["count", NaNs],
            )
            .sort_index()
            # .reset_index()
            .pipe(ut.flatten_cols)
            .pipe(catchstate, "df")
            .loc[df["count | " + self.dims.y] == 0, :]
        )

        if len(empties) != 0:
            warnings.warn(
                f"#! These groups contain no values: {empties.columns}\n"
                f"   Seaborn works, but statistics might raise KeyError!",
                stacklevel=10000,
            )
            ut.pp(empties)

    # def groupby(self):
    #     """We nee our own groupby function that does not drop empty groups"""
    #     for

    def iter_rowcol(self, skip_empty=True):
        """
        A generator that iterates through data grouped by facets/row-col
        :return:
        """
        for factor in ut.ensure_list(self.factors_rowcol):
            pass

        # for name, df in self.data.groupby(self.factors_rowcol):
        #     yield name, df

    def iter_allgroups(self, skip_empty=True):
        """
        A generator that iterates through data grouped by row, col, hue and x
        :return:
        """
        for name, df in self.data.groupby(self.factors_all):
            yield name, df

    ### TRANSFORM  ..................................................................................................'''

    def transform(self, func: str | Callable, inplace=False):
        """Transforms the data, changes dv property"""

        default_trafofuncs = {
            "ln": np.log,
            "log2": np.log2,
            "log": np.log10,
            "log10": np.log10,
            "sqrt": np.sqrt,
            None: None,
            # "none": None,
        }

        assert func in default_trafofuncs or callable(
            func
        ), f"#! '{func}' should be callable OR one of {list(default_trafofuncs.keys())}"

        a = self if inplace else ut.copy_by_pickling(self)
        func = func if callable(func) else default_trafofuncs[func]

        def rename_y(y: str) -> str:
            if type(func) is str:
                return f"{y}_({func})"
            elif callable(func):
                return f"{y}_({func.__name__})"

        def add_transform_col(
            df: "pd.DataFrame", y_raw: str, y_new: str
        ) -> "pd.DataFrame":
            nan_count_before = df[y_raw].isna().sum()
            if not y_new in df:
                df = ut.insert_after(df=df, after=y_raw, newcol_name=y_new, func=func)
            nan_count_after = df[y_new].isna().sum()
            if nan_count_before != nan_count_after:
                warnings.warn(
                    f"\n#! '{y_new}' has changed NaN-count ({nan_count_before} -> {nan_count_after}) after "
                    f"transformation!\n",
                    stacklevel=1000,
                )
            return df

        y_raw = a._y_untransformed
        y_new = rename_y(y=y_raw)
        a = a.set(y=y_new, inplace=inplace)
        a.data = add_transform_col(df=a.data, y_raw=y_raw, y_new=y_new)
        self.is_transformed = True
        self.transform_funcs.append(func)

        return a

    def reset_y(self, inplace=False):
        a = self if inplace else ut.copy_by_pickling(self)
        a = a.set(y=self._y_untransformed, inplace=inplace)
        self.is_transformed = False
        # self.transform_func = []  #* KEEP HISTORY OF TRANSFORMATION
        return a

    ### EDIT DIMS ..................................................................................................."""

    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> "Analysis":
        a = self if inplace else copy(self)

        # * NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply"""
        a.dims = a.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)
        # a.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg) # NOT WORKIN IN-CHAIN

        return a

        # newdims = self.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)
        #
        # if inplace:
        #     self.dims = newdims
        #     return self
        # else:
        #     newobj = copy(self)
        #     newobj.dims = newdims
        #     return newobj

    def set(
        self,
        dims: "Dims" | dict = None,
        y: str = None,
        x: str = None,
        hue: str = None,
        row: str = None,
        col: str = None,
        data: "pd.DataFrame" = None,
        transform: str | Callable = None,
        title: str = None,
        inplace=False,
        verbose=True,
    ) -> "Analysis":
        """Redefines values of Analysis.dims (y,x,hue,row,col) and also title,"""

        ### HANDLE COPY"""
        a = self if inplace else ut.copy_by_pickling(self)

        if not dims is None:
            if isinstance(dims, dict):
                a.dims = Dims(**dims)
            else:
                a.dims = dims

        if any((y, x, hue, row, col)):
            kwargs = {
                key: arg
                for key, arg in zip(
                    ("y", "x", "hue", "row", "col"), (y, x, hue, row, col)
                )
                if not arg is None
                ### WE ALSO ALLOW "none" TO REMOVE A DIMENSION
            }

            ### NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply"""
            a.dims = a.dims.set(inplace=inplace, verbose=verbose, **kwargs)
            # a.dims.set(inplace=inplace, verbose=verbose, **kwargs) # NOT WORKIN IN-CHAIN

        if not data is None:
            a.data = data
            # print(data)

        if not transform is None:
            a.transform(func=transform)

        if not title is None:
            a.title = title

        return a

    ### An Alias

    def update_analysis(
        self,
        y=None,
        x=None,
        hue=None,
        row=None,
        col=None,
        data: "pd.DataFrame" = None,
        transform: str | Callable = None,
        title: str = None,
        inplace=False,
        verbose=True,
    ) -> "Analysis":
        """Same as `self.update_analysis`, just with inplace=True"""

        return self.set(
            y=y,
            x=x,
            hue=hue,
            row=row,
            col=col,
            data=data,
            title=title,
            inplace=inplace,
            verbose=verbose,
        )

    def pool_facet(self, facet: str | tuple | list, inplace=False):
        """
        Sets e.g. col to None so that dataset is no longer splitted into subdata
        :param facet:
        :return:
        """
        a = self if inplace else copy(self)

        facet = ut.ensure_tuple(facet)
        facets = (
            "row",
            "col",
            "hue",
            # self.dims.row, self.dims.col, self.dims.hue
        )
        for f in facet:
            assert (
                f in facets
            ), f"#! {facet} should have been one of {[f for f in facets if not f is None]}"

        kws = {f: "none" for f in facet}
        return a.set(**kws)

    ### EXPERIMENTAL ################################################################################################"""
    # def pool_levels(self):
    #     """pools certain levels within a factor together"""
    #
    #
    #     # SEE 3_VWELLS-ADHESION/21ZF_NEW
    #     DF["Time_pool"] = (
    #         DF["Time"]
    #         .cat.add_categories("1-3")
    #         .mask(DF["Time"].isin(["1", "2", "3"]),"1-3")
    #         .cat.remove_unused_categories()
    #         .cat.reorder_categories(new_categories=["1-3", "24"], ordered=True))
    #
    #     '''!! WE ALSO NEED NEW COLUMN FOR SUBJECTS!!!'''
    #     DF["Replicate | Time"] = (
    #         DF[['Replicate', 'Time']]
    #         .astype(str)
    #         .apply(" | ".join, axis=1)# .astype("category")
    #     )
    #
    #     pass
