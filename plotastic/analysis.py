from __future__ import annotations  # for type hinting my Class type for return values
import numpy as np
import pandas as pd
from copy import copy
from typing import Dict, List, Callable, TYPE_CHECKING

# from dataclasses import dataclass
# import importlib

import warnings
from requests import get

import seaborn as sns
from scipy.stats import skew as skewness
import matplotlib.pyplot as plt

# import markurutils.UTILS as ut
# from markurutils.builtin_types import printable_dict
# from markurutils.filer import Filer
import markurutils as ut
from dims import Dims


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
        
        self._NaN, self._empty_groups = self.get_empties(verbose=False) #* shows empty groups

        self.is_transformed = False
        self.transform_funcs = []  # * HISTORY OF TRANSFORMATIONS
        self._y_untransformed = self.dims.y  # * STORE IT SO WE CAN RESET IT AFTER TRANSFORMATION

        self._factors_all = None  # [x, hue, row, col] defined in dims
        self._factors_xhue = None
        self._factors_rowcol = None
        self._levels = None

        self._vartypes = None  # * Categorical or Continuous? (Nominal? Ordinal? Discrete? Contiuous?)

        self.filer = ut.Filer(title=title)
        # if importlib.util.find_spec("pyrectories"):
        #    from pyrectories import Filer
        # else:
        # from markurutils.filer import Filer

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

    ### FACTORS .....................................................................................................'''

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
        F = (self.dims.row, self.dims.col, self.dims.hue, self.dims.x)
        self._factors_all = [e for e in F if (not e is None)]
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

    #### LEVELS ......................................................................................................'''

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
        """
        :return: [("row_lvl1", "row_lvl2"), ("col_lvl1", "col_lvl2"), ("hue_lvl1", "hue_lvl2"), ("hue_lvl1", "hue_lvl2")]
        """
        return [tuple(l) for l in self.levels.values() if not l is None]

    @property
    def levels_xhue_flat(self) -> tuple:
        """
        :return: (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
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
        """Has ROW (not the factor) as keys!
        :return: {"ROW":[ror_l1, row_l2, ...], "COL":[c_l1, c_l2, ...], "HUE":[...], "X":[...]}
        """

        D = self.levels
        return {
            "ROW": D.get(self.dims.row),
            "COL": D.get(self.dims.col),
            "HUE": D.get(self.dims.hue),
            "X": D.get(self.dims.x),
        }

    # ... DESCRIBE DATA ...............................................................................................'''

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

    def get_empties(self, verbose=False):
        """Returns a list of all groups/facets that are missing in the Dataframe.
        :return: _description_"""
        ### Make complete df with all possible groups/facets and with factors as index
        df = self.data_ensure_allgroups().set_index(self.factors_all)

        # * Rows with single NaNs
        allNaN_df = df[df.isna().all(axis=1)]
        # * Rows with only NaNs (these are completely missing in self.data)
        hasNaN_df: "pd.DataFrame" = df[df.isna().any(axis=1) & ~df.isna().all(axis=1)]

        allNaN_list = allNaN_df.index.to_list()
        hasNaN_list = hasNaN_df.index.to_list()

        if verbose:
            if len(allNaN_df) > 0:
                print(
                    "❗️ Among all combinations of selected factors, these groups/facets are missing in the Dataframe:"
                )
                print(allNaN_list, "\n")
            else:
                print(
                    "✅ All combinations of selected factors are present in the Dataframe"
                )

            if len(hasNaN_df) > 0:
                print("!!! These groups/facets contain single NaNs:")
                print(hasNaN_list, "\n")
                ut.pp(hasNaN_df)
            else:
                print("✅ No groups with single NaNs")

        ### Return DF for NaNs, and list for empty groups
        return allNaN_df, hasNaN_list

    @property
    def empty_groups(self) -> list:
        self._empty_groups = self.get_empties()[0]
        return

    @property
    def NaNs(self) -> "pd.DataFrame":
        self._NaNs = self.get_empties()[1]
        return self._NaNs

    # ... Iterate through DATA  .......................................................................................................'''

    def data_ensure_allgroups(self) -> pd.DataFrame:
        """df.groupby() skips empty groups, so we need to ensure that all groups are present in the data.
        :return: _description_
        :rtype: _type_
        """

        # * Index with incomplete set of groups
        reindex_DF = self.data.set_index(self.factors_all)
        index_old = reindex_DF.index

        # * Index with complete set of groups
        index_new = pd.MultiIndex.from_product(
            self.levels_tuples, names=self.factors_all
        )

        ### Construct empty DF but with complete 'index' (index made out of Factors)
        empty_DF = pd.DataFrame(
            index=pd.Index.difference(index_new, index_old),
            columns=self.columns_not_factor,
        )
        # * Fill empty DF with data
        newDF = pd.concat([empty_DF, reindex_DF]).sort_index().reset_index()
        return newDF

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

    # ... TRANSFORM  ..................................................................................................'''

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

    # ... SETTERS ..................................................................................................."""

    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> "Analysis":
        a = self if inplace else copy(self)

        # * NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply
        a.dims = a.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)

        return a

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
        title: str = None, # type: ignore
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
                # * WE ALSO ALLOW "none" TO REMOVE A DIMENSION
            }

            # * NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply"""
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

    # ... EXPERIMENTAL ################################################################################################"""
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
