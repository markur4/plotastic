"""We utilize List of all dims/ factors in a specific orders to group
and index the data into (fully) facetted datagroups. This allows easier
implementation of ... - ... displaying missing levels of the last factor
(x or hue) per group - ... connecting datapoints of the same subject
across x and hue levels - (... iterating through all datagroups for
statistics) 
"""

# %%

# from pprint import pprint
# from IPython.display import display

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotastic as plst
from plotastic.dimensions.subject import Subject
from plotastic.utils import utils as ut


from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis


# %%


class HierarchicalDims(Subject):
    def __init__(self, **kws):
        super().__init__(**kws)

    @property
    def _factors_hierarchical(self) -> list:
        """Return list of factors that are used for indexing the
        subjectdata. It places subjects before x and hue, which is
        useful to see which x and hue level are missing per subject"""
        factors = [
            self.dims.row,
            self.dims.col,
            self.subject,  # < Subject
            self.dims.x,
            self.dims.hue,
        ]
        ### Kick out Nones:
        return [f for f in factors if f is not None]

    @property
    def _factors_hierarchical_subjects_last(self) -> list:
        """Return list of factors that are used for indexing the
        subjectdata. It places subjects after x and hue, which is
        useful to see which subject is missing per x and hue level"""
        factors = [
            self.dims.row,
            self.dims.col,
            self.dims.x,
            self.dims.hue,
            self.subject,  # < Subject
        ]
        ### Kick out Nones:
        return [f for f in factors if f is not None]

    def data_hierarchicize(
        self,
        sort=True,
        subjects_last=False,
    ) -> pd.DataFrame:
        """Return Dataframe indexed by all factors containing only
        columns y and subjects"""

        ### Pick order of Hierarchy
        if subjects_last:
            factors = self._factors_hierarchical_subjects_last
        else:
            factors = self._factors_hierarchical

        ### Pick Data and set Index
        DF = self.data[factors + [self.dims.y]]
        DF = DF.set_index(factors)

        ### Sort
        if sort:
            DF = DF.sort_index()

        return DF

    def _iter__hlkey_df(
        self, sort=False, subject_last=False, by_lastdim=False
    ) -> Generator[tuple[tuple[str | int], pd.DataFrame], None, None]:
        """Iterate over data_hierarchical, return hierarchical levelkeys
        and dataframe"""
        ### Pick order of Hierarchy
        if subject_last:
            factors = self._factors_hierarchical_subjects_last
        else:
            factors = self._factors_hierarchical

        ### Remove last dim (x or hue)
        # > Otherwise we iterate over single rows
        if not by_lastdim:
            factors = factors[:-1]

        ### Pandas doesn't like grouping by length 1 tuples/lists
        if len(factors) == 1:
            factors = factors[0]

        for key, df in self.data_hierarchicize(
            sort=sort, subjects_last=subject_last
        ).groupby(factors):
            yield key, df

    def get_missing_lvls_from_last_factor(
        self,
        show_all=False,
        as_dict=False,
    ) -> pd.DataFrame | dict:
        """Return dataframe with missing levels per group. If show_all
        is False, only groups with missing levels are shown."""
        ### Reference for complete levels
        all_x_lvls = tuple(self.levels_dict_dim["x"])
        all_hue_lvls = tuple(self.levels_dict_dim["hue"])

        ### Collect Missing
        missing = {}
        for key, df in self._iter__hlkey_df():
            if self.dims.hue:
                hue_lvls = tuple(df.index.get_level_values(self.dims.hue))
                missing[key] = tuple(set(all_hue_lvls) - set(hue_lvls))
            else:
                x_lvls = tuple(df.index.get_level_values(self.dims.x))
                missing[key] = tuple(set(all_x_lvls) - set(x_lvls))

        ### Remove groups that didn't have any missing values
        if not show_all:
            # > Convert v to list so that resulting DataFrame has
            #' just one column
            missing = {k: [v] for k, v in missing.items() if v}

        ### Convert Result to DataFrame
        if not as_dict:
            missing = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    tuples=missing.keys(),
                    names=self._factors_hierarchical[:-1],
                ),
                data=missing.values(),
                columns=["missing levels"],
            ).sort_index()

        return missing


# %%
if __name__ == "__main__":
    # == Test Data =====================================================

    def make_testdata_paired_but_nosubject():
        ### Attention
        DF = sns.load_dataset("attention")
        dims = dict(y="score", x="attention", hue="solutions")
        DA1 = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
        DA1.test_pairwise(paired=False)

        ### qPCR
        DF, dims = plst.load_dataset("qpcr", verbose=False)
        #' DA2
        dims = dict(y="fc", x="gene", row="fraction", col="class")
        DA2 = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
        DA2.test_pairwise(paired=False)
        #' DA3
        dims = dict(y="fc", x="gene", hue="fraction", col="class")
        DA3 = plst.DataAnalysis(data=DF, dims=dims, verbose=False)
        DA3.test_pairwise(paired=False)

        return (DA1, DA2, DA3)

    def make_testdata():
        ### Attention
        DF = sns.load_dataset("attention")
        #' DA4 - no col, but hue
        dims = dict(y="score", x="attention", hue="solutions")
        DA4 = plst.DataAnalysis(
            data=DF, dims=dims, subject="subject", verbose=False
        )
        DA4.test_pairwise(paired=False)

        #' DA5 - no hue, but col
        dims = dict(y="score", x="solutions", col="attention")
        DA5 = plst.DataAnalysis(
            data=DF, dims=dims, subject="subject", verbose=False
        )
        DA5.test_pairwise(paired=True)

        ### qPCR
        DF, dims = plst.load_dataset("qpcr", verbose=False)
        #' DA6
        dims = dict(y="fc", x="gene", hue="fraction", col="class")
        DA6 = plst.DataAnalysis(
            data=DF, dims=dims, subject="subject", verbose=False
        )
        DA6.test_pairwise(paired=True)

        #' DA7 - with row
        dims = dict(y="fc", x="gene", row="fraction", col="class")
        DA7 = plst.DataAnalysis(
            data=DF, dims=dims, subject="subject", verbose=False
        )
        DA7.test_pairwise(paired=True)

        return DA4, DA5, DA6, DA7

    DA1, DA2, DA3 = make_testdata_paired_but_nosubject()
    DA4, DA5, DA6, DA7 = make_testdata()

    # %%
    ### Test when executed with DA doesn't have subject specified
    # DA1.subjectlist # > Gives error rightfully

    DA1.get_missing_lvls_from_last_factor()
    DA2.get_missing_lvls_from_last_factor()
    DA3.get_missing_lvls_from_last_factor()

    # DA1._subjects_get_XY() # > Gives error correctly
    # DA1.plot_connect_subjects() # > Gives error correctly

    # %%
    # DA1.catplot()

    # DA3.data_hierarchicize()
    # DA3.levels_get_missing()

    # %%
    # DA4.get_hierarchical_data(sorted=True)
    # DA4.get_hierarchical_data(sorted=True)
    DA6.data_hierarchicize(sort=True, subjects_last=True)
    # %%
    DA6.data_hierarchicize(sort=True, subjects_last=False)

    # %%
    # for subject, df in DA6.subjects_iter__subject_df:
    #     pprint(subject)
    #     pprint(df)
    #     print()

    # %%
    # DA4.subjects_get_missing()
    # DA5.subjects_get_missing()
    DA6.get_missing_lvls_from_last_factor()
    # %%
    # pprint(DA4.subjects_get_XY())
    # pprint(DA5.subjects_get_XY())
    # pprint(DA6.subjects_get_XY())
    DA4._subjects_get_XY()
    # DA5.subjects_get_XY()
    # DA6.subjects_get_XY().loc[("MMPs", slice(None), "MMP7"), :]
    # DF = DA6.subjects_get_XY()
    # DF[DF.index.get_level_values("class") == "Chemokines"].index
    # # DF.index

    # %%
    def plottest(self: plst.DataAnalysis, figsize=(2.5, 2), **plot_kws):
        (
            self.subplots(figsize=figsize)
            .fillaxes(
                kind="swarm",
                size=2,
                dodge=True,
            )
            .edit_y_scale_log(10)
            .plot_connect_subjects(**plot_kws)
            .annotate_pairwise()
        )
        return self

    plottest(DA4)
    plottest(DA5)
    plottest(DA6, figsize=(12, 4))
    plottest(DA7, figsize=(12, 12))
