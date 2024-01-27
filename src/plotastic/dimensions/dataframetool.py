#

# %% Imports

from re import L, T  # for type hinting my Class type for return values
from typing import Callable, Generator, Self, TYPE_CHECKING
import warnings

from copy import deepcopy

import pandas as pd

# import tabulate as tab

import numpy as np
import seaborn as sns
from scipy.stats import skew as skewness
import matplotlib.pyplot as plt

# import markurutils as ut
import plotastic.utils.utils as ut

# from plotastic.dimensions.dimsandlevels import DimsAndLevels
from plotastic.dimensions.hierarchical_dims import HierarchicalDims

from typing import Tuple

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

# %% Class DataFrameTool


class DataFrameTool(HierarchicalDims):
    # ==
    # == __init__ ======================================================

    def __init__(
        self,
        levels: list[tuple[str]] = None,
        **kws,
    ):
        """Adds pandas DataFrame related tools.

        :param levels:  If levels are specified, they will be compared \
                with the dataframe and columns will be set to ordered categorical type
                automatically, Defaults to None.
        :type levels: list[tuple[str]], optional
        :param verbose: Run object in verbose mode, which means it'll
            print a lot of feedback, Defaults to True.
        :type verbose: bool, optional
        """

        ### Inherit
        super().__init__(**kws)

        ### Transformations
        self.transformed = False
        self.transform_history = []  #' HISTORY OF TRANSFORMATIONS
        #' Store y so we can reset it after transformations
        self._y_untransformed = self.dims.y

        ### Make Categorical
        #' Levels are excluded levels, if unspecified
        self.user_levels = levels  # TODO: Rework this
        if levels:
            self._check_inputlevels_with_data(
                input_lvls=levels, verbose=False
            )
            self.input_levels = levels
            _ = self.data_categorize(verbose=False)

    # ==
    # ==  Make Levels Categorical ======================================

    def _make_catdict_from_input(
        self, input_lvls: list[list[str]], skip_notfound=True
    ) -> dict:
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
        #' This ensures that we can categorize only those columns whose levels were specified in the input
        # !! This also makes sure that those factors, that were completely mismatched, don't appear in the result
        catdict = {k: v for k, v in catdict.items() if len(v) > 0}
        return catdict

    def _count_matching_levels(self, input_lvls: list[list[str]]) -> int:
        """Counts how many levels match with the dataframe
        Args:
            levels (list[list[str]]): List of list of strings
        """
        catdict = self._make_catdict_from_input(input_lvls, skip_notfound=True)
        return sum([len(v) for v in catdict.values()])

    def _check_inputlevels_with_data(
        self,
        input_lvls: list[list[str]],
        verbose=True,
        strict=False,
    ) -> None:
        """Checks input levels with Dataframe and detects dissimilarities
        Args:
            all_lvls (list[list[str]]): List of lists of all levels. The order of the lists do not have to match that of the dataframe.
        """

        ### Compare with Dict:
        #' Keys:   Factors that have levels that match with INPUT
        #' Values: Levels from DATA
        #' -> {f1: [input_lvl1, input_lvl2], f2: [input_lvl1, input_lvl2], ...}
        #' This ensures that we can categorize only those columns whose levels were specified in the input
        catdict = self._make_catdict_from_input(
            input_lvls, skip_notfound=False
        )  #' dsfaadsf
        LVLS = {}
        for factor_from_input, lvls_from_input in catdict.items():
            if factor_from_input == "NOT_FOUND":
                LVLS[factor_from_input] = lvls_from_input
            else:
                LVLS[factor_from_input] = self.levels_dict_factor[
                    factor_from_input
                ]

        # LVLS = {factor: self.levels_factor_dict[factor] if not factor is 'NOT_FOUND' for factor in catdict.keys()}

        ### Scan for Matches and Mismatches
        matchdict = {}
        for factor, LVLs_fromCOL in LVLS.items():
            matchdict[factor] = (False, LVLs_fromCOL)  #' Initialize
            if factor == "NOT_FOUND":
                continue  # !! LVLs_fromCOL won't contain levels from data but actually from input
            for lvls in input_lvls:
                if ut.check_unordered_identity(
                    LVLs_fromCOL, lvls, ignore_duplicates=False
                ):
                    matchdict[factor] = (
                        True,
                        LVLs_fromCOL,
                    )  #' Update if match
                    break
        if verbose:
            self._print_levelmatches(input_lvls, matchdict)

    def _print_levelmatches(self, input_lvls, matchdict) -> None:
        """Prints the level matches and mismatches

        Args:
            matchdict (_type_): _description_
            strict (bool, optional): _description_. Defaults to False.

        Raises:
            AssertionError: _description_
        """
        #' List all unmatched levels
        problems, warnings = [], []
        if "NOT_FOUND" in matchdict.keys():
            # !! This is treated as a warning, since it might contain levels that fully exclude a factor
            mismatches = matchdict["NOT_FOUND"][1]
            warnings.append(mismatches)
            print(f"🟡 Levels mismatch:  {mismatches}")
        #' Check for mismatches per factor
        problems = []
        for factor, (match, LVLs_fromCOL) in matchdict.items():
            if not match and factor != "NOT_FOUND":
                problems.append(factor)
                print(
                    f"🟠 Levels incomplete: For '{factor}', your input does not cover all levels"
                )
        #' Give feedback how much was matched
        if self._count_matching_levels(input_lvls) == 0:
            print("🛑🛑🛑 Levels bad: No input matched with data")
        elif not problems and not warnings:
            print(
                "✅ Levels perfect: All specified levels were found in the data"
            )
        elif not problems:
            print("✅ Levels good: No partially defined factors")
        elif self._count_matching_levels(input_lvls) > 0:
            print("🆗 Levels ok: Some input was found")

        #' Search through input levels and data levels
        if problems:
            self._print_levelmatches_detailed(input_lvls)

    def _print_levelmatches_detailed(self, input_lvls) -> None:
        """Prints out detailed summary of level mismatches between user input and data

        Args:
            input_lvls (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.
        """

        # !! ALWAYS VERBOSE

        RJ = 17  #' Right Justification to have everything aligned nicely

        ### Make a catdict
        catdict = self._make_catdict_from_input(input_lvls)

        ### In DATA (but not in INPUT)
        print("\n   >> Searching levels that are in DATA but not in INPUT...")
        print("     ", f"LEVELS IN DATA: ".rjust(15), "DEFINED BY USER?")

        input_lvl_flat = ut.flatten(input_lvls)
        for factor, LVLs_from_COL in self.levels_dict_factor.items():
            for lvl_df in LVLs_from_COL:
                if (
                    not factor in catdict.keys()
                ):  #' When all levels from a factor are missing
                    message = f"<-- Undefined, like all levels from '{factor}'. This will be ignored"
                elif lvl_df in input_lvl_flat:  #' MATCH
                    message = "yes"
                else:  #' when factor was partially defined
                    message = f"<-- 🚨 UNDEFINED. Other levels from '{factor}' were defined, so this one will turn to NaNs!"
                #' Add '' to recognize hiding leading or trailing spaces
                lvl_df = f"'{lvl_df}'" if isinstance(lvl_df, str) else lvl_df
                print("     " + f"{lvl_df}: ".rjust(RJ), message)

        ### In INPUT (but not in DATA)
        print("\n   >> Searching levels that are in INPUT but not in DATA...")
        print("     ", f"USER INPUT: ".rjust(RJ), "FOUND IN DATA?")
        problems = []  #' Gather problems for error message
        for lvl in ut.flatten(input_lvls):
            #' Returns None if nothing found
            found: str | None = self.get_factor_from_level(lvl)
            #' Add '' to recognize hiding leading or trailing spaces
            lvl = f"'{lvl}'" if isinstance(lvl, str) else lvl
            if not found is None:
                message = f"Found in '{found}'"
            else:
                message = "<-- Not found. This input will be ignored"
            print("     ", f"{lvl}: ".rjust(RJ), message)
        print()

    def data_categorize(self, verbose=True) -> "DataFrameTool":
        """Categorize the data according to the levels specified in the constructor"""
        catdict = self._make_catdict_from_input(self.input_levels)

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
        return self

    # ==
    # == DESCRIBE DATA =================================================

    def catplot(self, kind="strip", **catplot_kws) -> sns.FacetGrid:
        """
        A simple seaborn catplot

        Returns:
            _type_: sns.FacetGrid
        """

        ### Handle kwargs
        kws = dict(
            kind=kind,
            data=self.data,
            height=2.5,
            **self.factors_as_dict,
            facet_kws=dict(despine=False),
        )
        kws.update(catplot_kws)

        g = sns.catplot(**kws)
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

    # ==
    # == Iterate through DATA ==========================================

    ### Iterate through FACETS =========================================

    def data_ensure_allgroups(self, factors=None) -> pd.DataFrame:
        """df.groupby() skips empty groups, so we need to ensure that all groups are present in the data.
        Takes Levels of factors. Returns a DataFrame with all possible combinations of levels as index.
        Args:
            factors (_type_, optional): Factors whose levels are to be used. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """

        factors = self.factors_all if factors is None else factors

        #' et columns with factors to index, yielding an index with incomplete keys
        reindex_DF = self.data.set_index(self.factors_all)
        index_old = reindex_DF.index

        #' Make index with complete set of keys
        #' If only one factor, we need to use pd.Index instead of pd.MultiIndex
        if self.factors_is_just_x:
            index_new = pd.Index(
                data=self.levels_dict_dim["x"], name=self.dims.x
            )
        else:
            index_new = pd.MultiIndex.from_product(
                iterables=self.levels_tuples, names=self.factors_all
            )
        # index_new = pd.MultiIndex(levels=self.levelkeys_rowcol, names=self.factors_all)

        ### Construct empty DF but with complete 'index' (index made out of Factors)
        #' We need to fill it with float("NaN"), since .isnull doesn't recognize np.nan
        empty_DF = pd.DataFrame(
            index=pd.Index.difference(index_new, index_old),
            columns=self.columns_not_factor,
        ).fillna(float("NaN"))

        #' Fill empty DF with data
        newDF = pd.concat([empty_DF, reindex_DF]).sort_index().reset_index()

        # ut.pp(empty_DF)
        # ut.pp(reindex_DF)
        # print(reindex_DF.index)
        # ut.pp(newDF)
        return newDF

    @property  #' >>> (R_l1, C_l1), df1 >>> (R_l1, C_l2), df2 >>> (R_l2, C_l1), df3 ...
    def data_iter__key_facet(self) -> Generator:
        """Returns: >> (R_l1, C_l1), df1 >> (R_l1, C_l2), df2 >> (R_l2, C_l1), df3 ..."""
        if self.factors_is_unfacetted:
            #' If no row or col, return all axes and data
            yield None, self.data_ensure_allgroups()  # !! Error for  df.groupby().get_group(None)

        else:
            #' Only fill in empty groups for row and col,
            grouped = self.data_ensure_allgroups().groupby(
                ut.ensure_list(self.factors_rowcol)
            )
            for key in self.levelkeys_rowcol:
                df = grouped.get_group(key)
                yield key, df

    @property  #' >>> (R_l1, C_l1), df1 >>> (R_l1, C_l2), df2 >>> (R_l2, C_l1), df3 ...
    def data_iter__key_facet_skip_empty(
        self,
    ) -> Generator[tuple[str | int, pd.DataFrame], None, None]:
        """Returns: >> (R_l1, C_l1), df1 >> (R_l1, C_l2), df2 >> (R_l2, C_l1), df3 ...
        Does not contain rows from empty groups"""
        if self.factors_rowcol is None:
            #' If no row or col, return all axes and data
            yield None, self.data  # !! Error for  df.groupby().get_group(None)

        else:
            grouped = self.data.groupby(ut.ensure_list(self.factors_rowcol))
            for key in self.levelkeys_rowcol:
                df = grouped.get_group(key)
                yield key, df

    @property  #' {key: df1, key2: df2, ...}
    def data_dict(self) -> dict:
        return dict(self.data_iter__key_facet)

    @property  #' {key: df1, key2: df2, ...}
    def data_dict_skip_empty(self) -> dict:
        return dict(self.data_iter__key_facet_skip_empty)

    #
    ### Iterate through LISTS of groups ================================

    @property  #' >>> (R_l1, C_l1, Hue_l1), df >>> (R_l1, C_l2, Hue_l1), df2 >>> ...
    def data_iter__key_groups(self):
        """Returns: >> (R_l1, C_l1,  Hue_l1), df >> (R_l1, C_l2, Hue_l1), df2 >> ...
        Yields Dataframes that lists groups
        """
        if not self.factors_is_just_x:
            for key, df in self.data_ensure_allgroups().groupby(
                self.factors_all_without_x
            ):
                yield key, df
        else:
            yield None, self.data_ensure_allgroups()

    @property  #' >>> (R_l1, C_l1, Hue_l1), df >>> (R_l1, C_l2, Hue_l1), df2 >>> ...
    def data_iter__key_groups_skip_empty(self):
        """Returns: >> (R_l1, C_l1,  Hue_l1), df >> (R_l1, C_l2, Hue_l1), df2 >> ...
        Yields Dataframes that lists groups
        """
        if not self.factors_is_just_x:
            for key, df in self.data.groupby(self.factors_all_without_x):
                yield key, df
        else:
            yield None, self.data

    #
    ### Iterate through GROUPS =========================================

    @property  #' >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__allkeys_group(self) -> Generator:
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ..."""
        for key, df in self.data_ensure_allgroups().groupby(self.factors_all):
            yield key, df

    @property  #' >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__allkeys_group_skip_empty(self) -> Generator:
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ...
        SKIPS EMPTY GROUPS!"""
        for key, df in self.data.groupby(self.factors_all):
            yield key, df

    #
    # == TRANSFORM =====================================================

    @staticmethod
    def _rename_y(y: str, func: str | Callable) -> str:
        """Renames the y column to reflect the transformation

        Args:
            y (str): _description_
            func (str): _description_

        Returns:
            str: _description_
        """
        if type(func) is str:
            return f"{y}_({func})"
        elif callable(func):
            return f"{y}_({func.__name__})"

    @staticmethod
    def _add_transform_col(
        df: pd.DataFrame,
        y_raw: str,
        y_new: str,
        func: Callable,
    ) -> pd.DataFrame:
        """Adds a column to the dataframe that contains the transformed data

        Args:
            df (pd.DataFrame): _description_
            y_raw (str): _description_
            y_new (str): _description_
            func (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        nan_count_before = df[y_raw].isna().sum()
        if not y_new in df:
            df = ut.insert_after(
                df=df, after=y_raw, newcol_name=y_new, func=func
            )
        nan_count_after = df[y_new].isna().sum()
        if nan_count_before != nan_count_after:
            warnings.warn(
                f"\n#! '{y_new}' has changed NaN-count ({nan_count_before} -> {nan_count_after}) after "
                f"transformation!\n",
                stacklevel=1000,
            )
        return df

    def transform_y(
        self, func: str | Callable, inplace=False
    ) -> "DataFrameTool | DataAnalysis":
        """DOC: Transforms the data, changes dv property"""

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

        A: "DataFrameTool" = self if inplace else deepcopy(self)
        func = func if callable(func) else default_trafofuncs[func]

        y_raw = A._y_untransformed
        y_new = A._rename_y(y=y_raw, func=func)

        ### Change dims.y to y-transform!
        A.dims.y = y_new
        # a = a.set(y=y_new, inplace=inplace)

        ### Add transformed column
        A.data = A._add_transform_col(
            df=A.data,
            y_raw=y_raw,
            y_new=y_new,
            func=func,
        )
        A.transformed = True
        A.transform_history.append(func)

        return A

    def transform_reset(self, inplace=False) -> "DataFrameTool | DataAnalysis":
        A: "DataFrameTool" = self if inplace else deepcopy(self)
        A = A.set(y=self._y_untransformed, inplace=inplace)
        A.transformed = False
        A.transform_history.append("reset")
        # self.transform_func = []  #* KEEP HISTORY OF TRANSFORMATION
        return A
