from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pandas as pd


class StatResults:
    DEFAULT_UNCHECKED = "NOT CHECKED"  # * If ASSUMPTION not tested,
    DEFAULT_UNTESTED = (
        "NOT TESTED"  # * If statistical test not tested (posthoc, omnibus)
    )
    DEFAULT_UNASSESSED = "NOT ASSESSED"  # * If not

    def __init__(self):
        ### Data Tables
        self.DF_normality: pd.DataFrame = self.DEFAULT_UNCHECKED
        self.DF_homoscedasticity: pd.DataFrame = self.DEFAULT_UNCHECKED
        self.DF_sphericity: pd.DataFrame = self.DEFAULT_UNCHECKED

        self.DF_omnibus_anova: pd.DataFrame = self.DEFAULT_UNTESTED
        self.DF_omnibus_rmanova: pd.DataFrame = self.DEFAULT_UNTESTED
        self.DF_omnibus_kruskal: pd.DataFrame = self.DEFAULT_UNTESTED
        self.DF_omnibus_friedman: pd.DataFrame = self.DEFAULT_UNTESTED
        self.DF_posthoc: pd.DataFrame = self.DEFAULT_UNTESTED
        self.DF_bivariate: pd.DataFrame = self.DEFAULT_UNTESTED

        ### Assessments = Summarizing results from multiple groups
        self._normal: bool = self.DEFAULT_UNASSESSED
        self._homoscedastic: bool = self.DEFAULT_UNASSESSED
        self._spherical: bool = self.DEFAULT_UNASSESSED

        self._parametric: bool = self.DEFAULT_UNASSESSED

    # == GETTERS AND SETTERS

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

    # == ASSESS ASSUMPTIONS

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
