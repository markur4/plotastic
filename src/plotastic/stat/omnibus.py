#
# %% imports

import warnings

import numpy as np
import pandas as pd
import pingouin as pg

# import markurutils as ut
import plotastic.utils.utils as ut

from plotastic.stat.assumptions import Assumptions

# %% Class Omnibus


class Omnibus(Assumptions):
    # ==
    # == __init__ ======================================================================
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

    # ==
    # == Helpers =======================================================================

    @staticmethod
    def _enhance_omnibus(DF: pd.DataFrame) -> pd.DataFrame:
        """Enhances the result DataFrame by adding additional columns

        :param DF: Result from omnibus_functions
        :type DF: pd.DataFrame
        :return: _description_
        :rtype: pd.DataFrame
        """
        ### Insert Star column right after "p-unc"
        stars = DF["p-unc"].apply(Omnibus._p_to_stars)
        DF.insert(DF.columns.get_loc("p-unc") + 1, "stars", stars)

        return DF

    def _ensure_more_than_one_sample_per_group(
        self,
        df: pd.DataFrame,
        facetkey: tuple = None,
    ) -> bool:
        """Returns an empty DataFrame if there are is only a single sample found within
        in all level combos. Useful to skip warning messages from pingouin.

        :param df: A facet of self.data
        :type df: pd.DataFrame
        :return: Either unchanged df or an empty DataFrame
        :rtype: pd.DtataFrame
        """

        ### Iterate through Sample groups within that facet
        results = []
        for levelkey, group in df.groupby(self.factors_xhue):
            if len(group) < 2:
                warnings.warn(
                    f"Skipping facet {facetkey}, because there is only one sample in {levelkey}",
                    RuntimeWarning,
                    stacklevel=3,  # ? Prints out function that calls this one (e.g. omnibus_anova) ?
                )
                results.append(False)
            else:
                results.append(True)

        ### Return True if all groups have more than one sample
        return all(results)

    # ==
    # == ANOVA =========================================================================

    def omnibus_anova(self, **user_kwargs) -> pd.DataFrame:
        """Performs an ANOVA (parametric, unpaired) on all facets of self.data

        :return: Result from pg.anova with row and column as MultiIndex
        :rtype: pd.DataFrame
        """
        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            between=self.factors_xhue,
            detailed=True,
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform ANOVA
        #' Skip empty groups
        aov_dict = {}
        for key, df in self.data_iter__key_facet_skip_empty:
            #' key = (row, col)
            aov = pg.anova(
                df, **kwargs
            )  # ? Doesn't seem to print annoying warnings
            aov_dict[key] = aov
        aov_DF = pd.concat(
            aov_dict, keys=aov_dict.keys(), names=self.factors_rowcol_list
        )

        ### Add extra columns
        aov_DF = self._enhance_omnibus(aov_DF)

        ### Save Result
        self.results.DF_omnibus_anova = aov_DF

        return aov_DF

    # ==
    # == RMANOVA =======================================================================

    def _omnibus_rm_anova_base(
        self,
        df: pd.DataFrame,
        facetkey: tuple,
        **kwargs,
    ) -> pd.DataFrame:
        """Handles Warnings of pg.rm_anova

        :param df: A facet of self.data
        :type df: pd.DataFrame
        :param facetkey: The key of the facet. Needed for warnings
        :type facetkey: tuple
        :return: Result from pg.rm_anova or empty DataFrame if there is only one sample
        :rtype: pd.DataFrame
        """
        ### Warn if there is only one sample in a group
        self._ensure_more_than_one_sample_per_group(df, facetkey)

        ### Perform RMANOVA
        # !! Pingouin slams you with warnings in a big loop
        # !! Trying best to redirect special cases, but still too many warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rmaov = pg.rm_anova(df, **kwargs)

        return rmaov

    def omnibus_rm_anova(
        self,
        **user_kwargs,
    ) -> pd.DataFrame:
        """Performs a repeated measures ANOVA (parametric, paired) on all facets of
        self.data


        :return: Result from pg.rm_anova with row and column as MultiIndex
        :rtype: pd.DataFrame
        """
        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            subject=self.subject,
            within=self.factors_xhue,
            detailed=True,
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform RMANOVA
        #' Skip empty groups
        rmaov_dict = {}
        for key, df in self.data_iter__key_facet_skip_empty:
            #' key = (row, col)
            rmaov = self._omnibus_rm_anova_base(df, facetkey=key, **kwargs)
            rmaov_dict[key] = rmaov
        rmaov_DF = pd.concat(
            rmaov_dict, keys=rmaov_dict.keys(), names=self.factors_rowcol_list
        )
        ### Add extra columns
        rmaov_DF = self._enhance_omnibus(rmaov_DF)

        ### Save Result
        self.results.DF_omnibus_rmanova = rmaov_DF

        return rmaov_DF

    # ==
    # == Kruskal-Wallis ================================================================

    def omnibus_kruskal(self, **user_kwargs) -> pd.DataFrame:
        """Performs a Kruskal-Wallis test (non-parametric, unpaired) on all facets of
        self.data


        :return: Result from pg.kruskal with row and column as MultiIndex
        :rtype: pd.DataFrame
        """
        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            between=self.dims.x,
            detailed=True,
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform Kruskal-Wallis
        #' pg.Kruskal takes only a single factor
        #' Skip empty groups
        kruskal_dict = {}
        for key, df in self.data_iter__key_groups_skip_empty:
            #' key = (row, col, hue)
            kruskal = pg.kruskal(df, **kwargs)
            kruskal_dict[key] = kruskal
        kruskal_DF = pd.concat(
            kruskal_dict,
            keys=kruskal_dict.keys(),
            names=self.factors_all_without_x,
        )
        ### Add extra columns
        kruskal_DF = self._enhance_omnibus(kruskal_DF)

        ### Save Result
        self.results.DF_omnibus_kruskal = kruskal_DF

        return kruskal_DF

    # ==
    # == Friedman ======================================================================

    def omnibus_friedman(self, **user_kwargs) -> pd.DataFrame:
        """Performs a Friedman test (non-parametric, paired) on all facets of self.data

        :return: Result from pg.friedman with row and column as MultiIndex
        :rtype: pd.DataFrame
        """
        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            subject=self.subject,
            within=self.dims.x,
            # detailed=True, # !! pg.friedman doesn't have this option
        )
        kwargs.update(user_kwargs)  #' Add user kwargs

        ### Perform Friedman
        #' pg.friedman takes only a single factor
        #' Skip empty groups
        friedman_dict = {}
        for key, df in self.data_iter__key_groups_skip_empty:
            #' key = (row, col, hue)
            friedman = pg.friedman(df, **kwargs)
            friedman_dict[key] = friedman
        friedman_DF = pd.concat(
            friedman_dict,
            keys=friedman_dict.keys(),
            names=self.factors_all_without_x,
        )
        ### Add extra columns
        friedman_DF = self._enhance_omnibus(friedman_DF)

        ### Save Result
        self.results.DF_omnibus_friedman = friedman_DF

        return friedman_DF


# !!
# !! end class

# %% Test Omnibus

if __name__ == "__main__":
    from plotastic.example_data.load_dataset import load_dataset

    DF, dims = load_dataset("fmri")
    DF, dims = load_dataset("qpcr")

    # %% CHECK pingouin ANOVA
    kwargs = dict(data=DF, dv=dims["y"], detailed=True)

    aov = pg.anova(between=[dims["x"], dims["hue"]], **kwargs)
    rmaov = pg.rm_anova(
        within=[dims["x"], dims["hue"]], subject="subject", **kwargs
    )
    kruskal = pg.kruskal(between=dims["hue"], **kwargs)

    # %% Make DataAnalysis

    DA = Omnibus(data=DF, dims=dims, subject="subject", verbose=True)

    # %% There's a problem with the Data: Only 1 sample in MMP and MACS

    ### Sort by xhue
    df2 = DF[(DF["class"] == "MMPs") & (DF["method"] == "MACS")].sort_values(
        ["gene", "fraction"]
    )
    len(df2)  #' 24
    levelkeys2 = df2.set_index([dims["x"], dims["hue"]]).index.unique()
    DA._ensure_more_than_one_sample_per_group(df2)
    # DA._plot_dendrogram_from_levelkeys(levelkeys2)

    pg.rm_anova(
        data=df2,
        dv=dims["y"],
        within=[dims["x"], dims["hue"]],
        detailed=True,
        subject="subject",
    )
    #

    # %% Check stuff

    aov = DA.omnibus_anova()
    rmaov = DA.omnibus_rm_anova()
    kruskal = DA.omnibus_kruskal()
    friedman = DA.omnibus_friedman()

    # %% Check Kruskal
