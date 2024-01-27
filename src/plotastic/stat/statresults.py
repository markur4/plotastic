# %% Imports
from typing import TYPE_CHECKING

from pathlib import Path

import pandas as pd

if TYPE_CHECKING:
    import pandas as pd

# %% class StatResults


class StatResults:
    # ==
    # == DEFAULTS ======================================================================
    # fmt: off
    DEFAULT_UNCHECKED = "NOT CHECKED"   #' If ASSUMPTION not tested,
    DEFAULT_UNTESTED = "NOT TESTED"     #' If statistical test not tested (posthoc, omnibus)
    DEFAULT_UNASSESSED = "NOT ASSESSED" #' If not
    # fmt: on

    # ==
    # == INIT ==========================================================================
    def __init__(self):
        ### Data Tables
        self.DF_normality: pd.DataFrame | str = self.DEFAULT_UNCHECKED
        self.DF_homoscedasticity: pd.DataFrame | str = self.DEFAULT_UNCHECKED
        self.DF_sphericity: pd.DataFrame | str = self.DEFAULT_UNCHECKED

        self.DF_omnibus_anova: pd.DataFrame | str = self.DEFAULT_UNTESTED
        self.DF_omnibus_rmanova: pd.DataFrame | str = self.DEFAULT_UNTESTED
        self.DF_omnibus_kruskal: pd.DataFrame | str = self.DEFAULT_UNTESTED
        self.DF_omnibus_friedman: pd.DataFrame | str = self.DEFAULT_UNTESTED
        self.DF_posthoc: pd.DataFrame | str = self.DEFAULT_UNTESTED
        self.DF_bivariate: pd.DataFrame | str = self.DEFAULT_UNTESTED

        ### Assessments = Summarizing results from multiple groups
        self._normal: bool | str = self.DEFAULT_UNASSESSED
        self._homoscedastic: bool | str = self.DEFAULT_UNASSESSED
        self._spherical: bool | str = self.DEFAULT_UNASSESSED

        self._parametric: bool | str = self.DEFAULT_UNASSESSED

    # ==
    # == Summarize Results =============================================================

    @property
    def as_dict(self) -> dict:
        d = dict(
            ### Assumptions
            normality=self.DF_normality,
            homoscedasticity=self.DF_homoscedasticity,
            sphericity=self.DF_sphericity,
            ### Omnibus
            anova=self.DF_omnibus_anova,
            rm_anova=self.DF_omnibus_rmanova,
            kruskal=self.DF_omnibus_kruskal,
            friedman=self.DF_omnibus_friedman,
            ### Posthoc
            posthoc=self.DF_posthoc,
            ### Bivariate
            bivariate=self.DF_bivariate,
        )

        ### Remove untested
        d = {k: v for k, v in d.items() if not isinstance(v, str)}

        return d

    def __iter__(self) -> tuple[str, pd.DataFrame]:
        for test_name, DF in self.as_dict.items():
            yield test_name, DF

    # ==
    # == GETTERS AND SETTERS ===========================================================

    @property
    def normal(self):
        if self._normal == self.DEFAULT_UNASSESSED:
            self._normal = self.assess_normality()
        return self._normal

    @normal.setter
    def normal(self, value: bool):
        print(f"#! Defining normality as {value}!")
        self._normal = value

    @property
    def parametric(self):
        if self._parametric == self.DEFAULT_UNASSESSED:
            self._parametric = self.assess_parametric()
        return self._parametric

    @parametric.setter
    def parametric(self, value: bool):
        print(f"#! Defining parametric as {value}!")
        self._parametric = value

    # ==
    # == ASSESS ASSUMPTIONS ============================================================

    def assess_normality(self, data) -> bool:
        """Uses result from normality test for each group and decides if data should be considered normal or not"""
        assert (
            self.DF_normality is not self.DEFAULT_UNCHECKED
        ), "Normality not tested yet"
        raise NotImplementedError
        self.normal = stats.normaltest(data)[1] > 0.05

    def assess_parametric(self):
        """Uses results from normality, homoscedasticity and sphericity tests to decide if parametric tests should be used"""
        self.parametric = self.normal and self.homoscedastic and self.spherical
        return self.parametric

    # ==
    # == EXPORT ========================================================================

    def save(self, fname: str | Path = "plotastic_results", verbose=True) -> None:
        """Exports all statistics to one excel file. Different sheets for different
        tests

        :param out: Path to save excel file, optional (default="")
        :type out: str, optional
        """
        ### Construct output path
        fname = Path(fname).with_suffix(".xlsx")

        ### Init writer for multiple sheets
        writer = pd.ExcelWriter(fname, engine="xlsxwriter")
        workbook = writer.book

        ### Iterate through results
        for test_name, DF in self.as_dict.items():
            worksheet = workbook.add_worksheet(test_name)  #' Make sheet
            writer.sheets[test_name] = worksheet  #' Add sheet name to writer
            DF.to_excel(writer, sheet_name=test_name)  #' # Write DF to sheet

        ### Save
        writer.close()
        
        ### Tell save location
        if verbose:
            print(f"Saved results to {fname.resolve()}")


# !!
# !! end class

# %% test it
# if __name__ == "__main__":

#     # %% Load Data, make DA, fill it with stuff
#     from plotastic.example_data.load_dataset import load_dataset
#     DF, dims = load_dataset("qpcr")
#     # DA = DataAnalysis(DF, dims)
#     # DA.test_pairwise()
