import numpy as np

from plotastic.dimensions.dataframetool import DataFrameTool
from plotastic.stat.statresults import StatResults


class StatTest(DataFrameTool):
    # == Class Attribute ===============================================================

    # * Alpha
    ALPHA = 0.05

    @classmethod
    def set_alpha(cls, value: float) -> None:
        cls.ALPHA = value

    # * Alpha Tolerance: Will still print out result if it nearly crossed alpha level.
    ALPHA_TOLERANCE = 0.075

    @classmethod
    def set_alpha_tolerance(cls, value: float) -> None:
        cls.ALPHA_TOLERANCE = value

    #
    #
    # == __init__=======================================================================
    def __init__(self, **dataframetool_kwargs):
        super().__init__(**dataframetool_kwargs)

        ### Composition
        self.results = StatResults()

    #
    #
    # == Helper functions ==============================================================

    @staticmethod
    def _p_to_stars(fl: float, alpha=0.05):
        # if mpl.get_backend() == "module://mplcairo.macosx":
        #     s = "★"
        # else:
        #     s= "*"
        s = "*"
        # s = "★"

        assert type(alpha) in [
            float,
        ], f"#! Alpha was type{alpha}, float required"
        a = alpha
        # use other stars ☆  ★ ★ ★   ٭★☆✡✦✧✩✪✫✬✭✮✯✰✵✶✷✸✹⭑⭒✴︎
        if a / 1 < fl:
            stars = "ns"
        elif a / 1 >= fl > a / 5:
            stars = s
        elif a / 5 >= fl > a / 50:
            stars = s * 2
        elif a / 50 >= fl > a / 500:
            stars = s * 3
        elif a / 500 >= fl:
            stars = s * 4
        else:
            stars = float("NaN")

        # display p-values that are between 0.05-0.06 not as stars, but show them
        if a * 1.4 >= fl > a:  # -0.01
            stars = round(fl, 3)  # Report p-values if they're 0.05 -0.06.

        return stars

    @staticmethod
    def _effectsize_to_words(fl: float, t=(0.01, 0.06, 0.14, 0.5)):
        if t[0] > fl:
            effectSize = "No Effect"
        elif t[0] <= fl < t[1]:
            effectSize = "Small"
        elif t[1] <= fl < t[2]:
            effectSize = "Medium"
        elif t[2] <= fl < t[3]:
            effectSize = "Large"
        elif t[3] <= fl:
            effectSize = "Huge"
        else:
            effectSize = float("NaN")
        return effectSize
