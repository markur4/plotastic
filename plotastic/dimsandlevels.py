# !
# %% Imports

from __future__ import annotations

# from nis import cat

# from operator import index, le
from re import L  # for type hinting my Class type for return values
from typing import Dict, Generator, List, Callable, TYPE_CHECKING

from copy import copy
from itertools import product

# from dataclasses import dataclass
# import importlib

# import warnings

# from requests import get

# import numpy as np
import pandas as pd

# import seaborn as sns
from scipy.stats import skew as skewness
import matplotlib.pyplot as plt

# from zmq import has

# import markurutils.UTILS as ut
# from markurutils.builtin_types import printable_dict
# from markurutils.filer import Filer
import markurutils as ut

# from old.dataanalysis import DataAnalysis
from plotastic.dims import Dims

# %% Utils
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


# %% Class Analysis ..........................................................

class DimsAndLevels:
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
            D["data_ensure_allgroups"] = (
                D["data_ensure_allgroups"].shape,
                list(D["data_ensure_allgroups"].columns),
            )
        if "fig2nd" in D:
            D["fig2nd"] = f"{len(D['fig2nd'])} axes"
        # if "ax" in D:
        #     D["ax"] = len(self.ax)

        return ut.printable_dict(D=D, start_message=f"{type(self)}: ")

    # ... INIT ......................

    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict | Dims,
    ) -> DimsAndLevels:
        """_summary_

        Args:
            data (pd.DataFrame): Pandas dataframe, long-format!
            dims (dict | Dims): Dims object storing x, y, hue, col, row.
            verbose (bool, optional): Warns User of empty groups. Defaults to False.
            levels (list[tuple[str]], optional): If levels are specified, they will be compared \
                with the dataframe and columns will be set to ordered categorical type automatically. Defaults to None.
            som (dict[str, str], optional): Scales of measurements. NOT IMPLEMENTED YET. Defaults to None.

        Returns:
            Analysis: _description_
        """
        self.data = data
        self.dims = dims if type(dims) is Dims else Dims(**dims)

    #
    #
    # ... List FACTORS .....................................................................................................'''

    @property  # * [row, hue, x] (dims may be missing)
    def factors_all(self) -> list[str]:
        F = (self.dims.row, self.dims.col, self.dims.hue, self.dims.x)
        return [e for e in F if (not e is None)]
    
    # @property
    # def factors_all_include_none(self) -> list[str]:
    #     F = (self.dims.row, self.dims.col, self.dims.hue, self.dims.x)
    #     return [e for e in F]

    @property  # * {"y": dims.y, "x": dims.x, "hue": dims.hue, "col": dims.col, "row": dims.row}
    def factors_as_dict(self) -> dict:
        """
        gets the dimensions in forms of a dictinary to be passed onto seaborn functions
        :return:
        {"y": self.dims.y, "x": self.dims.x,"hue": self.dims.hue, "col": self.dims.col, "row": self.dims.row}
        :rtype: dict
        """
        return self.dims.asdict(incl_None=False)
        # return {dim: factor for dim, factor in self.dims.asdict().items() if not factor is None}

    @property  # * [col1, col7, col8]
    def columns_not_factor(self) -> list[str]:
        return [c for c in self.data.columns if c not in self.factors_all]

    @property
    def factors_xhue(self) -> str | list[str]:
        if self.dims.hue:
            xhue = [self.dims.x, self.dims.hue]
        else:
            xhue = self.dims.x
        return xhue

    # @property
    # def factors_huex(self) -> str | tuple[str]:
    #     """
    #     When within/between/factors has two elements, it DOES matter in which order you post them.
    #     Usually, multiple t-tests don't always compare two groups coming from some combination of factors and levels, since they assume that these combinations don't make sense, and sometimes they really don't.
    #     We still want to be able to get every combination and do this by simply switching the order of factors.
    #     This allows us to compare e.g. not just every 'treatment type' per 'timepoint', but also every 'timepoint' per 'treatment type'.
    #     """
    #     if self.dims.hue:
    #         xhue = (self.dims.hue, self.dims.x)
    #     else:
    #         xhue = self.dims.x
    #     return xhue

    # @property
    # def factors_xhue_huex(self) -> tuple[str] | tuple[tuple[str]]:
    #     """Makes sure that x is not present twice in the list if no hue is present"""
    #     return tuple(set(self.factors_xhue, self.factors_huex))

    # @property
    # def factors_xhue_list(self) -> list[str]:
    #     if self.dims.hue:
    #         xhue = [self.dims.x, self.dims.hue]
    #     else:
    #         xhue = [self.dims.x]
    #     return xhue

    @property
    def factors_rowcol(self) -> str | list[str] | None:
        if self.dims.row and self.dims.col:
            rowcol = [self.dims.row, self.dims.col]
        elif self.dims.row:
            rowcol = self.dims.row
        elif self.dims.col:
            rowcol = self.dims.col
        else:
            rowcol = None
        return rowcol

    @property
    def factors_rowcol_list(self) -> list[str]:
        if self.dims.row and self.dims.col:
            rowcol = [self.dims.row, self.dims.col]
        elif self.dims.row:
            rowcol = [self.dims.row]
        elif self.dims.col:
            rowcol = [self.dims.col]
        else:
            rowcol = [""]
        return rowcol

    # ... Properties of FACTORS  .......................

    @property  # * True, False
    def is_just_x(self) -> bool:
        return not self.dims.row and not self.dims.col and not self.dims.hue

    @property
    def is_just_xand_hue(self) -> bool:
        return not self.dims.row and not self.dims.col

    @property  # * {"f1": "continuous", "f2": "category",}
    def factors_types(self) -> dict:
        """Returns: {"f1": "continuous", "f2": "category",}"""
        D = dict()
        for factor in self.factors_all:
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
        return D

    @property
    def factors_categoric(self):
        """Includes only columns that were defined as nominal or ordinal"""
        raise NotImplementedError

    #
    # ... Retrieve FACTORS .................................

    # * input: Hue -> "smoke"
    def getfactors_from_dim(
        self, putative_factors: str | list[str,]
    ) -> str | list[str]:
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

    def get_factor_from_level(self, level: str, ret_notfound=None):
        """Gets the factor from a level"""
        for factor, levels in self.levels_dict_factor.items():
            if level in levels:
                return factor
        ### If nothing returned, nothing was wound
        if ret_notfound is None:
            return None  # * We use this to check if the level is in the data
        elif ret_notfound == "raise":
            raise AssertionError(f"#! Level '{level}' not found in data.")

    # def get_rank_from_level(self, level: str):
    #     """Gets the factor from a level"""
    #     for rank, levels in self.hierarchy.items():
    #         if level in levels:
    #             return rank

    #
    # ... LEVELS ......................................................................................................'''

    def get_levels_from_column(self, colname: str) -> list[str]:
        """Returns: [lvl1, lvl2]"""
        S = self.data[colname]
        # if S.dtype.name == "category":
        if isinstance(S.dtype, pd.api.types.CategoricalDtype):
            return S.cat.categories.to_list()
        else:
            return S.unique().tolist()

    @property  # * {"f1": [lvl1, lvl2], "f2": [lvl1, lvl2],}
    def levels_dict_factor(self) -> dict:
        """Returns: {"f1": [lvl1, lvl2], "f2": [lvl1, lvl2],}"""
        return {
            factor: self.get_levels_from_column(colname=factor)
            for factor in self.factors_all
        }
    

    @property  # * {"row":[row_l1, row_l2, ...], "col":[c_l1, c_l2, ...], "hue":[...], "x":[...]}
    def levels_dict_dim(self) -> dict:
        """Returns: {"row":[row_l1, row_l2, ...], "col":[c_l1, c_l2, ...], "hue":[...], "x":[...]}"""
        D = self.levels_dict_factor
        return {
            "row": D.get(self.dims.row, ""),
            "col": D.get(self.dims.col, ""),
            "hue": D.get(self.dims.hue, ""),
            "x": D.get(self.dims.x),
        }

    @property  # * [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2), (hue_lvl1, hue_lvl2), (x_lvl1, x_lvl2)]
    def levels_tuples(self) -> list[tuple]:
        """Returns: [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2), (hue_lvl1, hue_lvl2), (x_lvl1, x_lvl2)]"""
        return [tuple(l) for l in self.levels_dict_factor.values() if not l is None]

    @property  # * [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2) ]
    def levels_tuples_rowcol(self):
        """Returns: [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2) ]"""
        return [
            tuple(l)
            for k, l in self.levels_dict_factor.items()
            if (not l is None) and (k in ut.ensure_list(self.factors_rowcol))
        ]

    @property  # * (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
    def levels_xhue_flat(self) -> tuple:
        """Returns: (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)"""
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

    @property  # * (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
    def levels_rowcol_flat(self) -> tuple:
        """Returns: (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)"""
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

    #
    # ... Properties of Levels .......................................................

    @property
    def len_rowlevels(self) -> int:
        if not self.dims.row is None:
            return len(self.levels_dict_factor[self.dims.row])
        else:
            return 1  # * Used by subplots, we need minimum of one row

    @property
    def len_collevels(self) -> int:
        if not self.dims.col is None:
            return len(self.levels_dict_factor[self.dims.col])
        else:
            return 1  # * Used by subplots, we need minimum of one col

    #

    #
    # ... SETTERS ..................................................................................................."""

    # !
    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> DimsAndLevels:
        a = self if inplace else copy(self)

        # * NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply
        a.dims = a.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)

        return a

    # !
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
        title: str = None,  # type: ignore
        inplace=False,
        verbose=True,
    ) -> DimsAndLevels:
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
            a.y_transform(func=transform)
        if not title is None:
            a.title = title
        return a

    def update_dims(
        self,
        y=None,
        x=None,
        hue=None,
        row=None,
        col=None,
        data: "pd.DataFrame" = None,
        # transform: str | Callable = None,
        title: str = None,
        inplace=False,
        verbose=True,
    ) -> DimsAndLevels:
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


# !
# !
# !    #* ######################################################################################


# %%

# DF, dims = ut.load_dataset("tips")
# dims = dict(y="tip", x="sex", hue="day", col="smoker", row="time")

# DA = Analysis(
#     data=DF,
#     dims=dims,
#     # subject="day",
#     verbose=True,
# )


# %% automatic testing


def tester(DF, dims):
    A = DimsAndLevels(data=DF, dims=dims, verbose=True)  # .switch("x", "col")


dimses = [
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

# DF, dims = ut.load_dataset("tips")
# for dim in dimses:
#     print("\n !!!", dim)
#     tester(DF, dim)


# %% Prototypes
# ! not working
"""! I tried implementing generators that iterate through the data vy looping through the list of
levelkeys and then picking out the datawindow using grouped.get_group(key). 
This is cool since I can make a function that uses skip=True/False as an argument, but generators don't accept arguments that easily
Also, I don't think I need it
"""
# def skip():
#     grouped = DA.data.groupby(DA.factors_all)
#     for key in DA.levelkeys_all:
#         try:
#             df = grouped.get_group(key)
#         except KeyError:
#             continue
#         yield key, df


# def notskip():
#     grouped = DA.data.groupby(DA.factors_all)
#     for key in DA.levelkeys_all:
#         try:
#             df = grouped.get_group(key)
#         except KeyError:
#             # * Associate key with respective factor
#             factor_and_keys = {DA.get_factor_from_level(lvl): lvl for lvl in key}
#             factor_and_keys_df = pd.DataFrame(factor_and_keys, index=[0])

#             empty_df = pd.DataFrame(columns=DA.data.columns)

#             # * Append to empty df
#             df = empty_df.append(factor_and_keys_df, ignore_index=True)

#         yield key, df


# def data_iter__key_allgroups(skip_empty_groups=False):
#     if skip_empty_groups:
#         return skip
#     else:
#         return notskip


# # for key, df in data_iter__key_allgroups(skip_empty_groups=True):
# #     print(key, df)
# #     print()

# for key, df in skip():
#     print(key, df)
#     print()

# for key, df in notskip():
#     print(key, df)
#     print()
