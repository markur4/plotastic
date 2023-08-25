#

# %% Imports

from re import L  # for type hinting my Class type for return values
from typing import Callable, Generator, List, TYPE_CHECKING
import warnings
from itertools import product

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew as skewness
import matplotlib.pyplot as plt

import markurutils as ut
from plotastic.dimsandlevels import DimsAndLevels

# %% Class DataFrameTool


class DataFrameTool(DimsAndLevels):
    # ... __init__ ...............................................................

    def __init__(
        self,
        levels: list[tuple[str]] = None,
        # levels_ignore: list[tuple[str]] = None,
        subject: str = None,
        # transform=None
        verbose=False,
        **dims_data_kws,
    ):
        """
        _summary_

        Args:
            data (pd.DataFrame): _description_
            dims (dict): _description_
            verbose (bool, optional): _description_. Defaults to False.
            fig (mpl.figure.Figure, optional): _description_. Defaults to None.
            axes (mpl.axes.Axes, optional): _description_. Defaults to None.

        Returns:
            PlotTool: _description_
        """
        ### Inherit from DimsAndLevels
        super().__init__(**dims_data_kws)

        ### Attributes
        # * to make columns categorical (and exclude levels, if unspecified)
        self.user_levels = levels
        # * for paired analysis
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
        g = sns.catplot(kind=kind, data=self.data, **self.factors_as_dict)
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
                "✅ Data complete: All combinations of levels from selected factors are present in the Dataframe"
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
        if self.factors_is_just_x:
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
    def data_iter__key_facet(self) -> Generator:
        """Returns: >> (R_l1, C_l1), df1 >> (R_l1, C_l2), df2 >> (R_l2, C_l1), df3 ..."""
        if self.factors_is_unfacetted:
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
    def data_iter__key_facet_skip_empty(self) -> Generator:
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
    def data_iter__allkeys_groups(self):
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ..."""
        for key, df in self.data_ensure_allgroups.groupby(self.factors_all):
            yield key, df

    @property  # * >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__allkeys_groups_skip_empty(self):
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
