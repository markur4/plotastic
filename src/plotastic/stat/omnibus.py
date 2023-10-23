#
# %% imports

import warnings

import numpy as np
import pandas as pd
import pingouin as pg

import markurutils as ut

from plotastic.stat.assumptions import Assumptions

# %% Class Omnibus


class Omnibus(Assumptions):
    # == __init__ ======================================================================
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

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

    def omnibus_anova(self, **user_kwargs) -> bool:
        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            between=self.factors_xhue,
            detailed=True,
        )
        kwargs.update(user_kwargs)  # * Add user kwargs

        ### Perform ANOVA
        # * Skip empty groups
        aov_dict = {}
        for key, df in self.data_iter__key_facet_skip_empty:
            # * key = (row, col)
            aov = pg.anova(df, **kwargs)  # ? Doesn't seem to print annoying warnings
            aov_dict[key] = aov

        aov_DF = pd.concat(aov_dict, keys=aov_dict.keys(), names=self.factors_rowcol)

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
        # ! Pingouin slams you with warnings in a big loop
        # ! Trying best to redirect special cases, but still too many warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rmaov = pg.rm_anova(df, **kwargs)

        return rmaov

    def omnibus_rm_anova(
        self,
        **user_kwargs,
    ) -> bool:
        ### Gather Arguments
        kwargs = dict(
            dv=self.dims.y,
            subject=self.subject,
            within=self.factors_xhue,
            detailed=True,
        )
        kwargs.update(user_kwargs)  # * Add user kwargs

        ### Perform RMANOVA
        # * Skip empty groups
        rmaov_dict = {}
        for key, df in self.data_iter__key_facet_skip_empty:
            # * key = (row, col)
            rmaov = self._omnibus_rm_anova_base(df, facetkey=key, **kwargs)
            rmaov_dict[key] = rmaov

        rmaov_DF = pd.concat(
            rmaov_dict, keys=rmaov_dict.keys(), names=self.factors_rowcol
        )

        return rmaov_DF
    
    


# !
# ! end class

# %% Test features

if __name__ == "__main__":
    from plotastic.example_data.load_dataset import load_dataset

    DF, dims = load_dataset("fmri")
    DF, dims = load_dataset("qpcr")

    # %% CHECK pingouin ANOVA
    aov = pg.anova(
        data=DF,
        dv=dims["y"],
        between=[dims["x"], dims["hue"]],
        detailed=True,
    )
    rmaov = pg.rm_anova(
        data=DF,
        dv=dims["y"],
        within=[dims["x"], dims["hue"]],
        subject="subject",
        detailed=True,
    )

    # %% Make DataAnalysis

    DA = Omnibus(data=DF, dims=dims, subject="subject", verbose=True)

    # %% Check ANOVA

    aov = DA.omnibus_anova()

    # %% There's a problem with the Data: Only 1 sample in MMP and MACS

    ### Sort by xhue
    df2 = DF[(DF["class"] == "MMPs") & (DF["method"] == "MACS")].sort_values(
        ["gene", "fraction"]
    )
    len(df2)  # * 24
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
    # %% Check RMANOVA
    
    rmaov = DA.omnibus_rm_anova()

    # %%
