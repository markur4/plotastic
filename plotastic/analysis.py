# !
# %% Imports

from __future__ import annotations
from nis import cat

from operator import index, le
from re import L  # for type hinting my Class type for return values
from typing import Dict, Generator, List, Callable, TYPE_CHECKING

from copy import copy
from itertools import product

# from dataclasses import dataclass
# import importlib

import warnings

# from requests import get

import numpy as np
import pandas as pd
import seaborn as sns
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
        subject: str = None,
        levels: list[tuple[str]] = None,
        # transform=None
        verbose=False,
        # levels_ignore: list[tuple[str]] = None,
        # som: dict[str, str] = None,
    ) -> Analysis:
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
        self.subject = subject

        # ### Initialize a dataframe that contains  groups for every combination of factor levels (with dv = N)
        # self.data_allgroups = self.data_ensure_allgroups

        ### Transformations
        self.is_transformed = False
        self.transform_history = []  # * HISTORY OF TRANSFORMATIONS
        # * Store y so we can reset it after transformations
        self._y_untransformed = self.dims.y

        ### Check for empties or missing group combinations
        if verbose:
            self.warn_about_empties_and_NaNs()
            if subject:
                self.warn_about_subjects_with_missing_data()

        ### Make Categorical
        if levels:
            self.check_inputlevels_with_data(input_lvls=levels, verbose=verbose)
            self.input_levels = levels
            self.data_categorize(verbose=verbose)

        # if levels_ignore:
        #     self.check_inputlevels_with_data(input_lvls=levels_ignore, verbose=verbose)

    #
    #
    # ... List FACTORS .....................................................................................................'''

    @property
    def factors_all(self) -> list[str]:
        F = (self.dims.row, self.dims.col, self.dims.hue, self.dims.x)
        return [e for e in F if (not e is None)]

    @property
    def factors_as_kwargs(self) -> dict:
        """
        gets the dimensions in forms of a dictinary to be passed onto seaborn functions
        :return:
        {"y": self.dims.y, "x": self.dims.x,"hue": self.dims.hue, "col": self.dims.col, "row": self.dims.row}
        :rtype: dict
        """
        return self.dims.asdict(incl_None=False)
        # return {dim: factor for dim, factor in self.dims.asdict().items() if not factor is None}

    @property
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
    def is_just_x(self) -> bool:
        return not self.dims.row and not self.dims.col and not self.dims.hue

    @property
    def is_just_xand_hue(self) -> bool:
        return not self.dims.row and not self.dims.col

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

    @property
    def factors_categoric(self):
        """Includes only columns that were defined as nominal or ordinal"""
        return

    #
    # ... Retrieve FACTORS .....................................................................................................'''

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

    def get_rank_from_level(self, level: str):
        """Gets the factor from a level"""
        for rank, levels in self.hierarchy.items():
            if level in levels:
                return rank

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
    # ... Properties of Factors and Levels ............................................................................................

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
    # ...  Make Levels Categorical...............................................................................

    def make_catdict_from_input(self, input_lvls: list[list[str]], skip_notfound=True):
        """Convert the lazy list input of levels into a dictionary with factors as keys and levels as values
        Args:
            levels (list[list[str]]): List of list of strings
            notfound (str, optional): What to do if a level is not found in the dataframe. Defaults to "skip".
        """

        ### Fill with Results
        catdict = {factor: [] for factor in self.factors_all}
        if not skip_notfound:
            catdict["NOT_FOUND"] = []
        for input_lvl in ut.flatten(input_lvls):
            factor = self.get_factor_from_level(input_lvl)
            if not factor is None:
                catdict[factor].append(input_lvl)
            else:
                if not skip_notfound:
                    catdict["NOT_FOUND"].append(input_lvl)

        ### Remove empties
        # * This ensures that we can categorize only those columns whose levels were specified in the input
        # ! This also makes sure that those factors, that were completely mismatched, don't appear in the result
        catdict = {k: v for k, v in catdict.items() if len(v) > 0}
        return catdict

    def count_matching_levels(self, input_lvls: list[list[str]]):
        """Counts how many levels match with the dataframe
        Args:
            levels (list[list[str]]): List of list of strings
        """
        catdict = self.make_catdict_from_input(input_lvls, skip_notfound=True)
        return sum([len(v) for v in catdict.values()])

    def check_inputlevels_with_data(
        self,
        input_lvls: list[list[str]],
        verbose=True,
        strict=False,
    ):
        """Checks input levels with Dataframe and detects dissimilarities
        Args:
            all_lvls (list[list[str]]): List of lists of all levels. The order of the lists do not have to match that of the dataframe.
        """

        ### Compare with Dict:
        # * Keys:   Factors that have levels that match with INPUT
        # * Values: Levels from DATA
        # * -> {f1: [input_lvl1, input_lvl2], f2: [input_lvl1, input_lvl2], ...}
        # * This ensures that we can categorize only those columns whose levels were specified in the input
        catdict = self.make_catdict_from_input(input_lvls, skip_notfound=False)
        LVLS = {}
        for factor_from_input, lvls_from_input in catdict.items():
            if factor_from_input == "NOT_FOUND":
                LVLS[factor_from_input] = lvls_from_input
            else:
                LVLS[factor_from_input] = self.levels_dict_factor[factor_from_input]

        # LVLS = {factor: self.levels_factor_dict[factor] if not factor is 'NOT_FOUND' for factor in catdict.keys()}

        ### Scan for Matches and Mismatches
        matchdict = {}
        for factor, LVLs_fromCOL in LVLS.items():
            matchdict[factor] = (False, LVLs_fromCOL)  # * Initialize
            if factor == "NOT_FOUND":
                continue  # ! LVLs_fromCOL won't contain levels from data but actually from input
            for lvls in input_lvls:
                if ut.check_unordered_identity(
                    LVLs_fromCOL, lvls, ignore_duplicates=False
                ):
                    matchdict[factor] = (True, LVLs_fromCOL)  # * Update if match
                    break
        if verbose:
            self._print_levelmatches(input_lvls, matchdict)

    def _print_levelmatches(self, input_lvls, matchdict):
        """Prints the level matches and mismatches

        Args:
            matchdict (_type_): _description_
            strict (bool, optional): _description_. Defaults to False.

        Raises:
            AssertionError: _description_
        """
        # * List all unmatched levels
        problems, warnings = [], []
        if "NOT_FOUND" in matchdict.keys():
            # ! This is treated as a warning, since it might contain levels that fully exclude a factor
            mismatches = matchdict["NOT_FOUND"][1]
            warnings.append(mismatches)
            print(f"🟡 Levels mismatch:  {mismatches}")
        # * Check for mismatches per factor
        problems = []
        for factor, (match, LVLs_fromCOL) in matchdict.items():
            if not match and factor != "NOT_FOUND":
                problems.append(factor)
                print(
                    f"🟠 Levels incomplete: For '{factor}', your input does not cover all levels"
                )
        # * Give feedback how much was matched
        if self.count_matching_levels(input_lvls) == 0:
            print("🛑🛑🛑 Levels bad: No input matched with data")
        elif not problems and not warnings:
            print("✅ Levels perfect: All specified levels were found in the data")
        elif not problems:
            print("✅ Levels good: No partially defined factors")
        elif self.count_matching_levels(input_lvls) > 0:
            print("🆗 Levels ok: Some input was found")

        # * Search through input levels and data levels
        if problems:
            self._print_levelmatches_detailed(input_lvls)

    def _print_levelmatches_detailed(self, input_lvls):
        """Prints out detailed summary of level mismatches between user input and data

        Args:
            input_lvls (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.
        """

        # ! ALWAYS VERBOSE

        RJ = 17  # * Right Justification to have everything aligned nicely

        ### Make a catdict
        catdict = self.make_catdict_from_input(input_lvls)

        ### In DATA (but not in INPUT)
        print("\n   >> Searching levels that are in DATA but not in INPUT...")
        print("     ", f"LEVELS IN DATA: ".rjust(15), "DEFINED BY USER?")

        input_lvl_flat = ut.flatten(input_lvls)
        for factor, LVLs_from_COL in self.levels_dict_factor.items():
            for lvl_df in LVLs_from_COL:
                if (
                    not factor in catdict.keys()
                ):  # * When all levels from a factor are missing
                    message = f"<-- Undefined, like all levels from '{factor}'. This will be ignored"
                elif lvl_df in input_lvl_flat:  # * MATCH
                    message = "yes"
                else:  # * when factor was partially defined
                    message = f"<-- 🚨 UNDEFINED. Other levels from '{factor}' were defined, so this one will turn to NaNs!"
                # * Add '' to recognize hiding leading or trailing spaces
                lvl_df = f"'{lvl_df}'" if isinstance(lvl_df, str) else lvl_df
                print("     " + f"{lvl_df}: ".rjust(RJ), message)

        ### In INPUT (but not in DATA)
        print("\n   >> Searching levels that are in INPUT but not in DATA...")
        print("     ", f"USER INPUT: ".rjust(RJ), "FOUND IN DATA?")
        problems = []  # * Gather problems for error message
        for lvl in ut.flatten(input_lvls):
            # * Returns None if nothing found
            found: str | None = self.get_factor_from_level(lvl)
            # * Add '' to recognize hiding leading or trailing spaces
            lvl = f"'{lvl}'" if isinstance(lvl, str) else lvl
            if not found is None:
                message = f"Found in '{found}'"
            else:
                message = "<-- Not found. This input will be ignored"
            print("     ", f"{lvl}: ".rjust(RJ), message)
        print()

    def data_categorize(self, verbose=True):
        """Categorize the data according to the levels specified in the constructor"""
        catdict = self.make_catdict_from_input(self.input_levels)

        if verbose:
            nans_before = self.data.isna().sum().sum()
            print("👉 Categorizing data...")
            print(f"    Applying these levels: {catdict}")
        self.data = ut.multi_categorical(self.data, catdict, verbose=verbose)

        if verbose:
            nans_after = self.data.isna().sum().sum()
            print(
                f"    NaNs before: {str(nans_before).rjust(5)} / {self.data.size} total cells"
            )
            print(
                f"    NaNs after:  {str(nans_after).rjust(5)} / {self.data.size} total cells, >> +{nans_after - nans_before} NaNs"
            )

    #
    # ... DESCRIBE DATA ...............................................................................................'''

    def catplot(self, kind="strip") -> sns.FacetGrid:
        """
        A simple seaborn catplot

        Returns:
            _type_: sns.FacetGrid
        """
        g = sns.catplot(kind=kind, data=self.data, **self.factors_as_kwargs)
        plt.show()
        return g

    def data_describe(self, verbose=False, plot=False) -> pd.DataFrame:
        ### Plot Data
        if plot:
            self.plot_quick

        ### Define Functions
        def NaNs(s: "pd.Series"):
            result = int(s.isna().sum())
            return result
            # if not result is None and result != 0:
            #     return result
            # else:
            #     return None

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
                    "mean",
                    "std",
                    Q1,
                    "median",
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

    def data_get_rows_with_NaN(self) -> pd.DataFrame:
        ### Make complete df with all possible groups/facets and with factors as index
        df = self.data_ensure_allgroups.set_index(self.factors_all)
        # * Pick only rows where some datapoints are missing, not all
        hasNaN_df: "pd.DataFrame" = df[df.isna().any(axis=1) & ~df.isna().all(axis=1)]
        return hasNaN_df

    def data_get_empty_groupkeys(self) -> list[str | tuple[str]]:
        ### Make complete df with all possible groups/facets and with factors as index
        df = self.data_ensure_allgroups.set_index(self.factors_all)
        # * Rows with only NaNs (these are completely missing in self.data)
        allNaN_df = df[df.isna().all(axis=1)]
        return allNaN_df.index.to_list()

    def warn_about_empties_and_NaNs(self) -> None:
        allNaN_list = self.data_get_empty_groupkeys()
        hasNaN_df = self.data_get_rows_with_NaN()

        if len(allNaN_list) > 0:
            print(
                "❗️ Data incomplete: Among all combinations of levels from selected factors, these groups/facets are missing in the Dataframe:"
            )
            # * Print all empty groups
            for key in allNaN_list:
                print(key)
        else:
            print(
                "✅ Data complete: All combinations of levels from selected factors are present in the Dataframe."
            )

        if len(hasNaN_df) > 0:
            print(
                "❗️ Groups incomplete: These groups/facets contain single NaNs: (Use .get_rows_with_NaN() to see them all):"
            )
            hasNaN_df.head(10)  # * Show df
            # ut.pp(hasNaN_df)
        else:
            print("✅ Groups complete: No groups with NaNs")

    def warn_about_subjects_with_missing_data(self) -> None:
        """Prints a warning if there are subjects with missing data"""

        ### Get numbers of samples for each subject
        counts_persubj = (
            self.data.groupby(self.subject)
            .count()[self.dims.y]
            .sort_values(ascending=False)
        )
        ### Retrieve subjects with missing data

        ### check if all subjects have the same number of samples
        if counts_persubj.nunique() > 1:
            print(
                f"❗️ Subjects incomplete: The largest subject contains {counts_persubj.max()} datapoints, but these subjects contain less:"
            )
            missing_data_df = counts_persubj[counts_persubj != counts_persubj.max()]
            print(missing_data_df)
        else:
            print("✅ Subjects complete: No subjects with missing data")

    # # ... Iterate through Data SKIPPING OF EMPTY GROUPS .................

    # def data_iterate_by_rowcol(
    #     self,
    # ) -> Generator[pd.DataFrame, None, None]:
    #     """Iterates through the data, yielding a DataFrame for each row/col combination.
    #     :yield: _description_
    #     :rtype: _type_
    #     """
    #     for key in self.levelkeys_rowcol:
    #         yield self.data_get_rowcol(key)

    #
    # ... Iterate through DATA  .......................................................................................................'''

    @property  # * [ (R_l1, C_l1, X_l1, Hue_l1), (R_l1, C_l2, X_l1, Hue_l1), (R_l2, C_l1, X_l1, Hue_l1), ... ]
    def levelkeys_all(
        self,
    ) -> list[tuple]:  # ! refactored from 'levelkeys' -> 'levelkeys_all'
        """Returns: [ (R_l1, C_l1, X_l1, Hue_l1), (R_l1, C_l2, X_l1, Hue_l1), (R_l2, C_l1, X_l1, Hue_l1), ... ]"""
        return [key for key in product(*self.levels_tuples)]

    @property  # * [ (R_l1, C_l1), (R_l1, C_l2), (R_l2, C_l1), ... ]
    def levelkeys_rowcol(self) -> list[tuple | str]:
        """Returns: [ (R_l1, C_l1), (R_l1, C_l2), (R_l2, C_l1), ... ]"""
        return [
            key
            if not len(key) == 1
            else key[0]  # * If only one factor, string is needed, not a tuple
            for key in product(*self.levels_tuples_rowcol)
        ]

    @property
    def data_ensure_allgroups(self) -> pd.DataFrame:
        """df.groupby() skips empty groups, so we need to ensure that all groups are present in the data.
        :return: _description_
        :rtype: _type_
        """

        # * Set columns with factors to index, yielding an index with incomplete keys
        reindex_DF = self.data.set_index(self.factors_all)
        index_old = reindex_DF.index

        # * Make index with complete set of keys
        # * If only one factor, we need to use pd.Index instead of pd.MultiIndex
        if self.is_just_x:
            index_new = pd.Index(data=self.levels_dict_dim["x"], name=self.dims.x)
        else:
            index_new = pd.MultiIndex.from_product(
                iterables=self.levels_tuples, names=self.factors_all
            )
        # index_new = pd.MultiIndex(levels=self.levelkeys_rowcol, names=self.factors_all)

        ### Construct empty DF but with complete 'index' (index made out of Factors)
        # * We need to fill it with float("NaN"), since .isnull doesn't recognize np.nan
        empty_DF = pd.DataFrame(
            index=pd.Index.difference(index_new, index_old),
            columns=self.columns_not_factor,
        ).fillna(float("NaN"))

        # * Fill empty DF with data
        newDF = pd.concat([empty_DF, reindex_DF]).sort_index().reset_index()

        # ut.pp(empty_DF)
        # ut.pp(reindex_DF)
        # print(reindex_DF.index)
        # ut.pp(newDF)
        return newDF

    @property  # * >>> (R_l1, C_l1), df1 >>> (R_l1, C_l2), df2 >>> (R_l2, C_l1), df3 ...
    def data_iter__key_facet(self) -> Generator[tuple, pd.DataFrame]:
        """Returns: >> (R_l1, C_l1), df1 >> (R_l1, C_l2), df2 >> (R_l2, C_l1), df3 ..."""
        if self.factors_rowcol is None:
            # * If no row or col, return all axes and data
            yield None, self.data_ensure_allgroups  # ! Error for  df.groupby().get_group(None)

        else:
            grouped = self.data_ensure_allgroups.groupby(
                ut.ensure_list(self.factors_rowcol)
            )
            for key in self.levelkeys_rowcol:
                df = grouped.get_group(key)
                yield key, df

    @property  # * >>> (R_l1, C_l1), df1 >>> (R_l1, C_l2), df2 >>> (R_l2, C_l1), df3 ...
    def data_iter__key_facet_no_empty(self) -> Generator[tuple, pd.DataFrame]:
        """Returns: >> (R_l1, C_l1), df1 >> (R_l1, C_l2), df2 >> (R_l2, C_l1), df3 ...
        Does not contain rows from empty groups"""
        if self.factors_rowcol is None:
            # * If no row or col, return all axes and data
            yield None, self.data  # ! Error for  df.groupby().get_group(None)

        else:
            grouped = self.data.groupby(ut.ensure_list(self.factors_rowcol))
            for key in self.levelkeys_rowcol:
                df = grouped.get_group(key)
                yield key, df

    @property  # * >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__key_allgroups(self):
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ..."""
        for key, df in self.data_ensure_allgroups.groupby(self.factors_all):
            yield key, df

    @property  # * >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__key_allgroups_no_empty(self):
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ...
        SKIPS EMPTY GROUPS!"""
        for key, df in self.data.groupby(self.factors_all):
            yield key, df

    #
    # ... TRANSFORM  ..................................................................................................'''

    def y_transform(self, func: str | Callable, inplace=False):
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
        self.transform_history.append(func)

        return a

    def y_reset(self, inplace=False):
        a = self if inplace else ut.copy_by_pickling(self)
        a = a.set(y=self._y_untransformed, inplace=inplace)
        self.is_transformed = False
        # self.transform_func = []  #* KEEP HISTORY OF TRANSFORMATION
        return a

    #
    # ... SETTERS ..................................................................................................."""

    # !
    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> Analysis:
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
    ) -> Analysis:
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

    def update_analysis(
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
    ) -> Analysis:
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
    A = Analysis(data=DF, dims=dims, verbose=True)  # .switch("x", "col")


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


# %%
