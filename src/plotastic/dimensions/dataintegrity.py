"""Tests to run when plst.DataAnalysis(verbose=True)"""
# %%
# == Imports ============================================================

import numpy as np
import pandas as pd

from plotastic.utils import utils as ut
from plotastic.dimensions.dataframetool import DataFrameTool

from typing import Tuple

# %%
# == Class DataIntegrity ===============================================


class DataIntegrity(DataFrameTool):
    """Tests to run when plst.DataAnalysis(verbose=True)"""

    def __init__(self, **dataframetool_kws) -> None:
        super().__init__(**dataframetool_kws)

    # ==
    # == COUNT GROUPS AND SAMPLESIZE ===================================

    def data_get_samplesizes(self) -> pd.Series:
        """Returns a DataFrame with samplesizes per group/facet"""
        #' Get samplesize per group
        samplesize_df = (
            self.data.groupby(self.factors_all)
            .count()[  #' Counts values within each group, will ignore NaNs -> pd.Series
                self.dims.y
            ]  #' Use y to count n. -> Series with x as index and n as values
            .sort_values(ascending=False)  #' Sort by samplesize
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
            df #' Full or partial facetted DataFrame 
            .groupby(self.dims.x)  #' Iterate through groups
            .count() #' ounts values within each group, will ignore NaNs -> pd.Series
            [self.dims.y]  #' Use y to count n. -> Series with x as index and n as values
            ) 
        # fmt: on
        return count

    def data_count_groups(self) -> int:
        """groups through all factors and counts the number of groups"""
        count = (
            self.data.groupby(self.factors_all)
            .count()  #' Counts values within each group, will ignore NaNs -> pd.Series
            .shape[
                0
            ]  #' gives the length of the series, which is the number of groups
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

        #' This is the same, should also count uniques.
        # # fmt: off
        # count = (
        #     df #' Full or partial facetted DataFrame
        #     .groupby(self.dims.x)  #' Iterate through groups
        #     .count() #' Count values within each group, will ignore NaNs -> pd.Series
        #     .shape[0]  #' gives the length of the series, which is the number of groups
        #     )
        # # fmt: on

        return count

    # ==
    # == FIND MISSING DATA==============================================

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
        #' Rows with only NaNs (these are completely missing in self.data)
        allNaN_df = df[df.isna().all(axis=1)]
        empty_group_keys = allNaN_df.index.to_list()

        ### Is only X not fully represented in each group?
        #' Every x has data from each other factor, but not every other Factor has data from each x
        #' e.g. with the qPCR data
        #' Make a Tree ..?

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
            M.append(ut.wrap_text("These are the first 5 rows with NaNs:"))
            df = hasNaN_df.head(5)  #' how df
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
        #' This is not a problem, but it's good to know
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
        # df = self.levels_combocount(normalize=False)  #' It's called by _levels_always_together()

        ### Get every levelkey that has the max value from combocount
        #' AT = always together
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

            #' Get subjects with missing data
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
        ut.print_separator("=", length=width)  #' fresh new line

    # ==
    # == FIND WELL CONNECTED FACTORS/LEVELS ============================

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
        )  #' Get combocount

        ### Max value of combocountdf should be the number of levelkeys found in Data
        len_levelcombos = df.max().max()

        # !! Not True if every level of all factors is found in every group
        # !! See plst.load_data("tips") for example
        # assert len_levelcombos == len(
        #     self.levelkeys
        # ), "Max value of combocount_df should be the number of levelkeys"

        ### Get every levelkey that has the max value from combocount
        #' These levels are guaranteed to be found together at leat once!
        # fmt: off
        AT_df: pd.DataFrame = (         #' at = always together
            df[df == len_levelcombos]   #' t max values
            .dropna(axis=0, how="all")  #' drop rows that are all NaN
            .dropna(axis=1, how="all")  #' drop columns that are all NaN
        )

        ### Retrieve the fully connected levels
        AT_levels = (
            AT_df
            .where(AT_df != np.nan) #' remove NaNs
            .stack()                #' convert to Series with MultiIndex from columns
            .index.to_list()
        )
        # fmt: on

        ### Check if the always together levels fully cover every level of a factor
        #' If so, that factor is a good candidate to facet the data by
        leveldict = (
            self.levels_dict_factor
        )  #' factor as key, levellist as value

        AT_flat = set([level for levels in AT_levels for level in levels])
        AT_factors = []
        for factor, levels in leveldict.items():
            if all([level in AT_flat for level in levels]):
                # print(f"{factor} is fully connected")
                AT_factors.append(factor)

        return AT_levels, AT_factors
