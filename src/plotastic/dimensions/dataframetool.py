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
from plotastic.dimensions.dimsandlevels import DimsAndLevels

from typing import Tuple

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

# %% Class DataFrameTool


class DataFrameTool(DimsAndLevels):
    # ==
    # == __init__ ======================================================================

    def __init__(
        self,
        levels: list[tuple[str]] = None,
        # levels_ignore: list[tuple[str]] = None,
        subject: str = None,
        # transform=None
        verbose=False,
        **dims_data_kws,
    ):
        """Adds pandas DataFrame related tools.

        :param levels:  If levels are specified, they will be compared \
                with the dataframe and columns will be set to ordered categorical type
                automatically, Defaults to None.
        :type levels: list[tuple[str]], optional
        :param subject: Column defining the subject, which connects samples as dependent
            (patient, donor, date etc.). If specified, paired tests will be prioritized,
            defaults to None.
        :type subject: str, optional
        :param verbose: Warns User of empty groups, defaults to False
        :type verbose: bool, optional
        """

        ### Inherit from DimsAndLevels
        super().__init__(**dims_data_kws)

        ### Attributes
        # * to make columns categorical (and exclude levels, if unspecified)
        self.user_levels = levels
        # * for paired analysis
        self.subject = subject
        if not subject is None:
            assert (
                subject in self.data.columns
            ), f"#! Subject '{subject}' not in columns, expected one of {self.data.columns.to_list()}"

        # ### Initialize a dataframe that contains  groups for every combination of factor levels (with dv = N)
        # self.data_allgroups = self.data_ensure_allgroups()

        ### Transformations
        self.transformed = False
        self.transform_history = []  # * HISTORY OF TRANSFORMATIONS
        # * Store y so we can reset it after transformations
        self._y_untransformed = self.dims.y

        ### Check for empties or missing group combinations
        self.factors_always_together = None
        if verbose:
            self.data_check_integrity()

        ### Make Categorical
        if levels:
            self._check_inputlevels_with_data(
                input_lvls=levels, verbose=verbose
            )
            self.input_levels = levels
            _ = self.data_categorize(verbose=verbose)

        # if levels_ignore:
        #     self.check_inputlevels_with_data(input_lvls=levels_ignore, verbose=verbose)

    # ==
    # ==  Make Levels Categorical ======================================================

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
        # * This ensures that we can categorize only those columns whose levels were specified in the input
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
        # * Keys:   Factors that have levels that match with INPUT
        # * Values: Levels from DATA
        # * -> {f1: [input_lvl1, input_lvl2], f2: [input_lvl1, input_lvl2], ...}
        # * This ensures that we can categorize only those columns whose levels were specified in the input
        catdict = self._make_catdict_from_input(
            input_lvls, skip_notfound=False
        )  # * dsfaadsf
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
            matchdict[factor] = (False, LVLs_fromCOL)  # * Initialize
            if factor == "NOT_FOUND":
                continue  # !! LVLs_fromCOL won't contain levels from data but actually from input
            for lvls in input_lvls:
                if ut.check_unordered_identity(
                    LVLs_fromCOL, lvls, ignore_duplicates=False
                ):
                    matchdict[factor] = (
                        True,
                        LVLs_fromCOL,
                    )  # * Update if match
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
        # * List all unmatched levels
        problems, warnings = [], []
        if "NOT_FOUND" in matchdict.keys():
            # !! This is treated as a warning, since it might contain levels that fully exclude a factor
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

        # * Search through input levels and data levels
        if problems:
            self._print_levelmatches_detailed(input_lvls)

    def _print_levelmatches_detailed(self, input_lvls) -> None:
        """Prints out detailed summary of level mismatches between user input and data

        Args:
            input_lvls (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.
        """

        # !! ALWAYS VERBOSE

        RJ = 17  # * Right Justification to have everything aligned nicely

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
    # == DESCRIBE DATA =========================================================

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
    # == COUNT GROUPS AND SAMPLESIZE ===========================================

    def data_get_samplesizes(self) -> pd.Series:
        """Returns a DataFrame with samplesizes per group/facet"""
        # * Get samplesize per group
        samplesize_df = (
            self.data.groupby(self.factors_all)
            .count()[  # * Counts values within each group, will ignore NaNs -> pd.Series
                self.dims.y
            ]  # * Use y to count n. -> Series with x as index and n as values
            .sort_values(ascending=False)  # * Sort by samplesize
        )
        return samplesize_df

    def data_count_n_per_x(self, df: pd.DataFrame) -> pd.Series:
        """Counts the number of non NaN entries within y columns in a dataframe,
        grouping only by the x variable (not hue!).

        :param df: Dataframe within a for loop, where df is a facetted dataframe and
            grouped by hue.
        :type df: pd.DataFrame
        :return: Samplesize, or count of non-NaN rows within y-column
        :rtype: pd.Series with x as index and n as values
        """
        # fmt: off
        count = (
            df # * Full or partial facetted DataFrame 
            .groupby(self.dims.x)  # * Iterate through groups
            .count() # * Counts values within each group, will ignore NaNs -> pd.Series
            [self.dims.y]  # * Use y to count n. -> Series with x as index and n as values
            ) 
        # fmt: on
        return count

    def data_count_groups(self) -> int:
        """groups through all factors and counts the number of groups"""
        count = (
            self.data.groupby(self.factors_all)
            .count()  # * Counts values within each group, will ignore NaNs -> pd.Series
            .shape[
                0
            ]  # * gives the length of the series, which is the number of groups
        )
        return count

    def data_count_groups_in_x(self, df: pd.DataFrame) -> int:
        """Counts the number of groups within x variable (Not considering grouping by
        hue).

        :param df: Dataframe. Preferrably within a for loop, where df is a facetted
            dataframe and grouped by hue.
        :type df: pd.DataFrame
        :return: Number of groups
        :rtype: int
        """
        count = len(df[self.dims.x].unique())

        # * This is the same, should also count uniques.
        # # fmt: off
        # count = (
        #     df # * Full or partial facetted DataFrame
        #     .groupby(self.dims.x)  # * Iterate through groups
        #     .count() # * Count values within each group, will ignore NaNs -> pd.Series
        #     .shape[0]  # * gives the length of the series, which is the number of groups
        #     )
        # # fmt: on

        return count

    # ==
    # == FIND MISSING DATA======================================================

    def data_get_rows_with_NaN(self) -> pd.DataFrame:
        ### Make complete df with all possible groups/facets and with factors as index
        df = self.data_ensure_allgroups().set_index(self.factors_all)
        #'' Pick only rows where some datapoints are missing, not all
        hasNaN_df: "pd.DataFrame" = df[
            df.isna().any(axis=1) & ~df.isna().all(axis=1)
        ]
        return hasNaN_df

    def data_get_empty_groupkeys(self) -> list[str | tuple[str]]:
        ### Make complete df with all possible groups/facets and with factors as index
        df = self.data_ensure_allgroups().set_index(self.factors_all)
        # * Rows with only NaNs (these are completely missing in self.data)
        allNaN_df = df[df.isna().all(axis=1)]
        empty_group_keys = allNaN_df.index.to_list()

        ### Is only X not fully represented in each group?
        # * Every x has data from each other factor, but not every other Factor has data from each x
        # * e.g. with the qPCR data
        # * Make a Tree ..?

        return empty_group_keys

    def _data_check_empty_groups(self) -> list[str]:
        ### Check for empty groups
        allNaN_list = self.data_get_empty_groupkeys()
        different_x_per_facet = 0

        M = []

        if len(allNaN_list) > 0:
            M.append(
                f"""❗️ DATA INCOMPLETE: Among all combinations of levels
                 from selected factors {self.factors_all}, groups/facets
                 are missing in the Dataframe.
                 """
            )
            M.append(
                """👉 Call
                 .data_get_empty_groupkeys() to see them all.
                 """
            )

        elif different_x_per_facet:
            M.append(
                f"""👌 DATA COMPLETE: Although each facet has
                      different x-levels, all combinations of levels
                      from selected factors {self.factors_all} are
                      present in the Dataframe."""
            )

        else:
            M.append(
                f"""✅ DATA COMPLETE: All combinations of levels from
                 selected factors are present in the Dataframe,
                 including x."""
            )

        return M

    def _data_check_rows_with_NaN(
        self,
    ) -> [list[str] | Tuple[list[str], pd.DataFrame]]:
        ### Detect rows with NaN
        hasNaN_df = self.data_get_rows_with_NaN()

        M = []
        df = pd.DataFrame()

        if len(hasNaN_df) > 0:
            M.append(
                """❗️ GROUPS INCOMPLETE: Groups/facets contain single
                NaNs. 👉 Call .get_rows_with_NaN() to see them all.
                """
            )
            M.append(self._wrap_text("These are the first 5 rows with NaNs:"))
            df = hasNaN_df.head(5)  # * Show df
        else:
            M.append("✅ GROUPS COMPLETE: No groups with NaNs.")

        if df.empty:
            return M
        else:
            return M, df

    def _data_check_equal_samplesizes(
        self,
    ) -> [list[str] | Tuple[list[str], pd.DataFrame]]:
        ### Detect equal samplesize among groups:
        # * This is not a problem, but it's good to know
        samplesize_df = self.data_get_samplesizes()
        groupcount = self.data_count_groups()
        ss_avg = round(samplesize_df.mean(), 1)
        ss_std = round(samplesize_df.std(), 1)

        M = []
        df = pd.DataFrame()

        if samplesize_df.nunique() > 1:
            M.append(
                f"""🫠 GROUPS UNEQUAL: Groups ({groupcount} total) have
                    different samplesizes (n = {ss_avg} ±{ss_std})."""
            )
            M.append("""👉 Call .data_get_samplesizes() to see them.""")
            M.append(
                """These are the 5 groups with the largest
                    samplesizes:"""
            )
            df = samplesize_df.head(5)
        else:
            M.append(
                f"""✅ GROUPS EQUAL: All groups ({groupcount} total) have
                    the same samplesize n = {ss_avg}."""
            )

        if df.empty:
            return M
        else:
            return M, df

    def _levels_combocounts_eval(self) -> list[str]:
        """Evaluates counts of how often each level appears with another
        level to describe the structure of the data

        :return: _description_
        :rtype: pd.DataFrame
        """
        ### Count how often each level appears with another leve
        # df = self.levels_combocount(normalize=False)  # * It's called by _levels_always_together()

        ### Get every levelkey that has the max value from combocount
        # * AT = always together
        AT_levels, AT_factors = self._levels_always_together()

        M = []

        if len(AT_factors) != 0:
            M.append(
                f"""🌳 LEVELS WELL CONNECTED: These Factors have levels
                 that are always found together: {AT_factors}. """
            )
            M.append(
                """👉 Call .levels_combocount() or .levels_dendrogram() to
                see them all."""
            )

        ### Set Attributes
        # self.levels_always_together = AT_levels # ? Not seeing where this is useful
        self.factors_always_together = AT_factors  # ? Could be useful?

        return M

    def _data_check_subjects_with_missing_data(
        self,
    ) -> [list[str] | Tuple[list[str], pd.DataFrame]]:
        """Prints a warning if there are subjects with unequal
        samplesizes
        """

        ### Get numbers of samples for each subject
        counts_persubj = (
            self.data.groupby(self.subject)
            .count()[self.dims.y]
            .sort_values(ascending=False)
        )

        M = []
        df = pd.DataFrame()

        ### Check if all subjects have the same number of samples
        if counts_persubj.nunique() > 1:
            M.append(
                f"""❗️ Subjects incomplete: The largest subject contains
                    {counts_persubj.max()} datapoints, but these
                    subjects contain less:"""
            )

            # * Get subjects with missing data
            df = counts_persubj[counts_persubj != counts_persubj.max()]
        else:
            M.append("✅ Subjects complete: No subjects with missing data")

        if df.empty:
            return M
        else:
            return M, df

    @staticmethod
    def _print_messages(
        messages: [list[str] | Tuple[list[str], pd.DataFrame]],
        width=80,
        indent="   ",
    ) -> None:
        """Prints a list of messages, where each element is either a
        string or a tuple of a string and a DataFrame"""

        wrap_kws = dict(
            width=width,
            width_first_line=width,
            indent=indent,
        )

        def wrap_message(M: list, **wrap_kws):
            ### Wrap the message into correct line length
            M = [ut.wrap_text(m, **wrap_kws) for m in M]

            ### Join with newlines
            M = f"\n{indent}".join(M)

            return M

        ### Print messages
        for message in messages:
            #'' No DataFrame passed
            if isinstance(message, list):
                print(wrap_message(message, **wrap_kws))
            #'' DataFrame is also passed
            elif isinstance(message, tuple):
                print(wrap_message(message[0], **wrap_kws))
                # print(tab.tabulate(message[1]))  # Dataframe
                ut.print_indented(message[1].to_markdown(), indent=indent)

    def data_check_integrity(self, width=79) -> None:
        """Prints information about Integrity of the data, including
        empty groups, rows with NaN, equal samplesizes and factors that
        are suitable for faceting.
        """

        ### Collect Messages (MES)
        MESSAGES = []

        MESSAGES.append(self._data_check_empty_groups())
        MESSAGES.append(self._data_check_rows_with_NaN())
        MESSAGES.append(self._data_check_equal_samplesizes())

        ### No need to evaluate level combos if there is just one factor
        if not self.factors_is_just_x:
            # _Identify factors that are always together
            # _This sets attribute self.factors_always_together
            MESSAGES.append(self._levels_combocounts_eval())
        if self.subject:
            MESSAGES.append(self._data_check_subjects_with_missing_data())

        ### Print
        ut.print_separator("=", length=width)
        print("#! Checking data integrity...")
        self._print_messages(MESSAGES, width=width)
        ut.print_separator("=", length=width)  # * fresh new line

    # ==
    # == FIND WELL CONNECTED FACTORS/LEVELS ====================================

    def _levels_always_together(
        self,
    ) -> tuple[list[tuple[str]], list[str]]:
        """Get every levelkey that has the max value from combocount
        These levels are very useful! They show which levels are
        guaranteed to be found together (at least once)! These factors
        that facet our data very reliably without missing groups, which
        is a good candidate to specify as hue for example

        :return: _description_
        :rtype: list
        """

        ### Count how often each level appears with another level
        df = self.levels_combocounts(
            normalize=False, heatmap=False
        )  # * Get combocount

        ### Max value of combocountdf should be the number of levelkeys found in Data
        len_levelcombos = df.max().max()

        # !! Not True if every level of all factors is found in every group
        # !! See plst.load_data("tips") for example
        # assert len_levelcombos == len(
        #     self.levelkeys
        # ), "Max value of combocount_df should be the number of levelkeys"

        ### Get every levelkey that has the max value from combocount
        # * These levels are guaranteed to be found together at leat once!
        # fmt: off
        AT_df: pd.DataFrame = (         # * at = always together
            df[df == len_levelcombos]   # * get max values
            .dropna(axis=0, how="all")  # * drop rows that are all NaN
            .dropna(axis=1, how="all")  # * drop columns that are all NaN
        )

        ### Retrieve the fully connected levels
        AT_levels = (
            AT_df
            .where(AT_df != np.nan) # * remove NaNs
            .stack()                # * convert to Series with MultiIndex from columns
            .index.to_list()
        )
        # fmt: on

        ### Check if the always together levels fully cover every level of a factor
        # * If so, that factor is a good candidate to facet the data by
        leveldict = (
            self.levels_dict_factor
        )  # * factor as key, levellist as value

        AT_flat = set([level for levels in AT_levels for level in levels])
        AT_factors = []
        for factor, levels in leveldict.items():
            if all([level in AT_flat for level in levels]):
                # print(f"{factor} is fully connected")
                AT_factors.append(factor)

        return AT_levels, AT_factors

    # ==
    # == Iterate through Data SKIPPING OF EMPTY GROUPS =================================

    # *def data_iterate_by_rowcol(
    #     self,
    # ) -> Generator[pd.DataFrame, None, None]:
    #     """Iterates through the data, yielding a DataFrame for each row/col combination.
    #     :yield: _description_
    #     :rtype: _type_
    #     """
    #     for key in self.levelkeys_rowcol:
    #         yield self.data_get_rowcol(key)

    # ==
    # == Iterate through DATA ==========================================================

    ### Iterate through FACETS =========================================================

    def data_ensure_allgroups(self, factors=None) -> pd.DataFrame:
        """df.groupby() skips empty groups, so we need to ensure that all groups are present in the data.
        Takes Levels of factors. Returns a DataFrame with all possible combinations of levels as index.
        Args:
            factors (_type_, optional): Factors whose levels are to be used. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """

        factors = self.factors_all if factors is None else factors

        # * Set columns with factors to index, yielding an index with incomplete keys
        reindex_DF = self.data.set_index(self.factors_all)
        index_old = reindex_DF.index

        # * Make index with complete set of keys
        # * If only one factor, we need to use pd.Index instead of pd.MultiIndex
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
            yield None, self.data_ensure_allgroups()  # !! Error for  df.groupby().get_group(None)

        else:
            # * Only fill in empty groups for row and col,
            grouped = self.data_ensure_allgroups().groupby(
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
            yield None, self.data  # !! Error for  df.groupby().get_group(None)

        else:
            grouped = self.data.groupby(ut.ensure_list(self.factors_rowcol))
            for key in self.levelkeys_rowcol:
                df = grouped.get_group(key)
                yield key, df

    @property  # * {key: df1, key2: df2, ...}
    def data_dict(self) -> dict:
        return dict(self.data_iter__key_facet)

    @property  # * {key: df1, key2: df2, ...}
    def data_dict_skip_empty(self) -> dict:
        return dict(self.data_iter__key_facet_skip_empty)

    #
    ### Iterate through LISTS of groups ================================================

    @property  # * >>> (R_l1, C_l1, Hue_l1), df >>> (R_l1, C_l2, Hue_l1), df2 >>> ...
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

    @property  # * >>> (R_l1, C_l1, Hue_l1), df >>> (R_l1, C_l2, Hue_l1), df2 >>> ...
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
    ### Iterate through GROUPS =========================================================

    @property  # * >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__allkeys_group(self) -> Generator:
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ..."""
        for key, df in self.data_ensure_allgroups().groupby(self.factors_all):
            yield key, df

    @property  # * >>> (R_l1, C_l1, X_l1, Hue_l1), df >>> (R_l1, C_l2, X_l1, Hue_l1), df2 >>> ...
    def data_iter__allkeys_group_skip_empty(self) -> Generator:
        """Returns: >> (R_l1, C_l1, X_l1, Hue_l1), df >> (R_l1, C_l2, X_l1, Hue_l1), df2 >> ...
        SKIPS EMPTY GROUPS!"""
        for key, df in self.data.groupby(self.factors_all):
            yield key, df

    #
    # == TRANSFORM =====================================================================

    @staticmethod
    def _rename_y(y: str, func: str) -> str:
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
        func: str,
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
        self, func: str | Callable, inplace=True
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
