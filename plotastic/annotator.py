#
# %% imports

import types
import warnings
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import markurutils as ut

from plotastic.posthoc import PostHoc
from plotastic.multiplot import MultiPlot
from plotastic.omnibus import Omnibus
from plotastic.bivariate import Bivariate

# %% class Annotator


class Annotator(MultiPlot, Omnibus, PostHoc, Bivariate):
    ### Define types that are allowed as elements within exclude/include arguments
    _TYPES_SELECTION = tuple([str, bool] + ut.NUMERICAL_TYPES)

    # ... ­­init__ .....................................................................

    def __init__(self, **dataframetool_kws):
        ### Inherit
        # * verbosity false, since each subclass can test its own DataFrame
        super().__init__(**dataframetool_kws)

    #
    #
    # ... GROUP SELECTION ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ... Check User Arguments of Datagroup Selection ..........................................

    # * [x1, {hue4:(x3,x4)}, {x2:(hue1,hue2)}, hue3]  OR  [x_or_hue, {x_or_hue: xhuepair} ] ETC.
    def _check_selected_xhue(self, xhue_selected: list) -> None:
        """ENFORCES THIS STRUCTURE:
        Dictionaries selects specific pairs. Everything else selects all pairs containing that element.
        [x1, {hue4:(x3,x4)}, {x2:(hue1,hue2)}, hue3]
        [x_or_hue, {x_or_hue: xhuepair} ]
        [x_or_hue, {x_or_hue: (x_or_hue2, x_or_hue2) } ]
        :param xhue_selected:
        :return:
        """

        ### Define allowed types for elements of xhue
        # * Dictionaries selects specific pairs, everything else selects all pairs containing that element.
        # * Levels of xhue levels can be specified as string, int bool and all numpy numerical types.
        types_allowed = tuple(list(self._TYPES_SELECTION) + [dict])
        types_specific = self._TYPES_SELECTION

        ### Retrieve Levels
        LVLs = self.levels_xhue_flat
        LVLdict = self.levels_dict_factor
        LVLs_X = LVLdict[self.dims.x]
        if self.dims.hue:
            LVLs_HUE = LVLdict[self.dims.hue]

        ### If hue or x is specified, nothing must be checked as the complete x or hue will be included or excluded
        if xhue_selected in ("hue", "x"):
            return None

        ### CHECK IF SELECTION IS CORRECT
        assert isinstance(xhue_selected, list), f"#! '{xhue_selected}' should be a list"
        for xhue in xhue_selected:
            assert isinstance(xhue, types_allowed), (
                f"#!  {xhue} ({ut.get_type(xhue)}) should be a string, numerical-like type, bool or a dictionary (with an x-/hue-level as key"
                f" and a tuple of two x-/hue-levels as value"
            )
            if isinstance(xhue, types_specific):
                assert (
                    xhue in LVLs
                ), f"#!  {xhue} ({ut.get_type(xhue)}) should be one of {LVLs}"
            if isinstance(xhue, dict):
                # * {x_or_hue: xhuepair }
                for x_or_hue_key, xhuepair in xhue.items():
                    assert x_or_hue_key in LVLs, (
                        f"#! The passed key  {x_or_hue_key} ({ut.get_type(x_or_hue_key)}) should be one of {LVLs_X} if values are in HUE-levels,"
                        f" or one of {LVLs_HUE} if values are in X-levels"
                    )
                    assert isinstance(xhuepair, tuple) and len(xhuepair) == 2, (
                        f"#!  {xhuepair}  ({ut.get_type(xhuepair)}) should be a tuple with two values from either {LVLs_X}"
                        f" or {LVLs_HUE}"
                    )
                    for x_or_hue_value in xhuepair:
                        # * {x_or_hue: (x_or_hue2, x_or_hue2) }
                        assert x_or_hue_value in LVLs, (
                            f"#! The passed value {x_or_hue_value} ({ut.get_type(x_or_hue_value)}) should be one of {LVLs_X}, if key is in HUE-levels),"
                            f" or one of {LVLs_HUE}, if key is in X-levels)"
                        )
                        if x_or_hue_key in LVLs_HUE:
                            assert (
                                not x_or_hue_value in LVLs_HUE
                            ), f"#! When {x_or_hue_key} is from HUE, '{xhuepair}' should be made of {LVLs_X}"
                        if x_or_hue_key in LVLs_X:
                            assert (
                                not x_or_hue_value in LVLs_X
                            ), f"#! When {x_or_hue_key} is from X, '{xhuepair}' should be made of {LVLs_HUE}"
        """ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^"""

    # * (row1, col2)  OR  row1  OR  col2
    def _check_selected_rowcol(self, rowcol_selected: tuple | str) -> None:
        """
        ENFORCES THIS STRUCTURE:
        Tuples are used to select facet if both row and col are specified. If not facetted, row and col levels may be specified as strings, numericals, bools, etc.
        (row1, col2) or row1
        :param rowcol_selected:
        :return:
        """
        ### Define expected types for elements of rowcol
        # * Tuples are used if both row and col are specified.
        # * Row and col levels may be specified as strings, numericals, bools, etc.
        types_allowed = tuple(self._TYPES_SELECTION + [tuple])
        types_specific = self._TYPES_SELECTION

        ### Retrieve levels
        LVLs = self.levels_rowcol_flat
        LVLs_ROW = self.levels_dict_dim["row"]
        LVLs_COL = self.levels_dict_dim["col"]

        ### Warn user to specify both row and col explicitly, if both are present
        if len(self.factors_rowcol_list) == 2 and isinstance(rowcol_selected, str):
            warnings.warn(
                f"#! When facetting with both row and col, it's safest to specify both row and col explicitly and not just one of them.",
                stacklevel=10000,
            )

        ### CHECK IF SELECTION IS CORRECT
        typ = ut.get_type(rowcol_selected)
        # * types_allowed
        assert isinstance(
            rowcol_selected, types_allowed
        ), f"#!  {rowcol_selected} ({typ}) should be a tuple, string, bool or numerical-like type"
        # * rowcol_selected = 'row1' or 'col2'
        if isinstance(rowcol_selected, types_specific):
            assert (
                rowcol_selected in LVLs
            ), f"#!  {rowcol_selected} ({typ}) should be one of {LVLs}"  # #
        # * rowcol_selected = '(row1, col2)'
        elif isinstance(rowcol_selected, tuple):
            assert (
                len(self.factors_rowcol_list) == 2
            ), f"#! A tuple was passed: {rowcol_selected}, but only one of row or col is present in the data: {self.factors_rowcol_list}"
            assert (
                len(rowcol_selected) == 2
            ), f"#! Facet-key  {rowcol_selected} ({typ}) should have two strings"
            row_or_col1, row_or_col2 = rowcol_selected[0], rowcol_selected[1]
            for row_or_col in (row_or_col1, row_or_col2):
                typ = ut.get_type(row_or_col)  # * Yes, this has to be that ugly
                # * If both row and col are present
                if LVLs_ROW and LVLs_COL:
                    message = f"#!  {row_or_col} ({typ}) should be one of ROW-levels {LVLs_ROW} if other element is in COL-levels; or one of COL-levels {LVLs_COL} if other element is in ROW-levels"
                # * If only row is present
                elif LVLs_ROW and not LVLs_COL:
                    message = f"#!  {row_or_col} ({typ}) should be one of ROW-levels {LVLs_ROW}"
                # * If only col is present
                elif not LVLs_ROW and LVLs_COL:
                    message = f"#!  {row_or_col} ({typ})  should be one of COL-levels {LVLs_COL}"
                assert row_or_col in LVLs, message

            if row_or_col1 in LVLs_ROW:
                assert (
                    not row_or_col2 in LVLs_ROW
                ), f"#! When {row_or_col1} is from ROW, {row_or_col2} should be one of {LVLs_COL}"
            if row_or_col1 in LVLs_COL:
                assert (
                    not row_or_col2 in LVLs_COL
                ), f"#! When {row_or_col1} is from COL, {row_or_col2} should be one of {LVLs_ROW}"
        """ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^"""

    def _check_include_exclude(
        self,
        exclude: dict = None,
        exclude_in_facet: dict = None,
        include: dict = None,
        include_in_facet: dict = None,
    ):
        """Check if the passed arguments are valid.

        Args:
            exclude (dict, optional): _description_. Defaults to None.
            exclude_in_facet (dict, optional): _description_. Defaults to None.
            include (dict, optional): _description_. Defaults to None.
            include_in_facet (dict, optional): _description_. Defaults to None.
        """

        ### Selection over all facets
        if not exclude is None:
            self._check_selected_xhue(exclude)
        if not include is None:
            self._check_selected_xhue(include)

        ### If facet-specific selection, check if row or col is specified
        if (not exclude_in_facet is None) or (not include_in_facet is None):
            """CHECK IF SELECTION IS CORRECT"""
            assert (
                not self.factors_rowcol is None
            ), f"#! Use 'filter_in_xhue' instead of 'filter_in_facet', if data is not facetted into row or col: {self.factors_others}"

        ### Selection in indivicual Facets
        # * {(row, col): [ x1,  {hue4:(x3,x4)},    hue3 ]}
        if not include_in_facet is None:
            assert isinstance(
                include_in_facet, dict
            ), f"#! exclude-/include_in_facet should be a dictionary, but {type(include_in_facet)} was passed"
            for rowcol_included, xhue_included in include_in_facet.items():
                self._check_selected_rowcol(rowcol_included)
                self._check_selected_xhue(xhue_included)
        if not exclude_in_facet is None:
            assert isinstance(
                exclude_in_facet, dict
            ), f"#! exclude-/include_in_facet should be a dictionary, but {type(exclude_in_facet)} was passed"
            for rowcol_excluded, xhue_excluded in exclude_in_facet.items():
                self._check_selected_rowcol(rowcol_excluded)
                self._check_selected_xhue(xhue_excluded)

    # ... Match user Arguments with Data .......................................................

    def _match_selected_xhue(
        self,
        S: "pd.Series",
        xhue_selected: list[str | dict] | str,
        true_value: str = "incl.",
    ) -> bool | str:
        """

        :param S:
        :param xhue_selected:
            [x1, {hue4:(x3,x4)}, {x2:(hue1,hue2)}, hue3]
            [x_or_hue, {x_or_hue: xhuepair} ]
            [x_or_hue, {x_or_hue: (x_or_hue2, x_or_hue2) } ]
        :param true_value:
        :return:
        """

        # levels_xhue = self.levels_xhue_flat
        # leveldict = self.levels

        ### Get pairs and flatten them if hue is present (("l1", "l2"),("l3","l4"))
        if self.dims.hue:
            PAIR = ut.flatten(S["pairs"])
        else:
            PAIR = S["pairs"]
        CROSS = S["cross"]

        if xhue_selected in ("x", "hue"):
            match = true_value if xhue_selected == CROSS else 0
            return match

        for xhue in xhue_selected:
            """xhue should be a string or a dictionary
            (with an x-/hue-level as key and a tuple of two x-/hue-levels as value)"""
            if isinstance(xhue, self._TYPES_SELECTION):
                """xhue should be one of levels_xhue"""
                match = true_value if xhue in PAIR else 0
                if match:
                    return match
                # print("\t", c, PAIR, xhue, match, bool(match))
            elif isinstance(xhue, dict):
                KEY = ut.get_duplicate(PAIR)
                """{x_or_hue: xhuepair }"""
                for x_or_hue_key, xhuepair in xhue.items():
                    """The passed key '{x_or_hue_key}' should be one of leveldict[self.dims.x]} if values are in HUE-levels,
                     or one of {leveldict[self.dims.hue]} if values are in X-levels.
                    The passed value '{xhuepair}' should be a tuple with two values from either leveldict[self.dims.x] or
                    leveldict[self.dims.hue]}"""
                    if x_or_hue_key != KEY:
                        match = 0
                    else:
                        for x_or_hue_value in xhuepair:
                            """{x_or_hue: (x_or_hue_key, x_or_hue_values) The passed value '{x_or_hue_values}' should be one of
                            leveldict[self.dims.x]}, (if key is in HUE-levels) or one of {leveldict[self.dims.hue]},
                            (if key is in X-levels)"""
                            match = true_value if x_or_hue_value in PAIR else 0
                            if match:
                                return match
                            # if x_or_hue in leveldict[self.dims.hue]:
                            #     '''When '{x_or_hue}' is from HUE, '{xhuepair}' should be made of {leveldict[self.dims.x]}'''
                            # if x_or_hue in leveldict[self.dims.x]:
                            #     '''When '{x_or_hue}' is from X, '{xhuepair}' should be made of {leveldict[self.dims.hue]}'''

        return match

    def _match_selected_rowcol(
        self,
        S: "pd.Series",
        rowcol_selection_dict: dict,
        true_value: str = "incl.",
    ) -> bool | str:
        """

        :param S:
        :param : This is used in df.apply(), so we iterate through rows, which are pandas Series
        :param rowcol_selection_dict:
        :param true_value:
        :return:
        """

        ### EXTRACT row AND col (FACTORS_OTHERS) FROM PH SERIES
        ROWCOL = tuple(S[self.factors_rowcol_list])

        match = np.nan
        for rowcol_selected, xhue_selected in rowcol_selection_dict.items():
            """{rowcol_selected} should be a tuple or string and one of {LVLs}"""  # #
            if isinstance(rowcol_selected, self._TYPES_SELECTION):
                match_rc = rowcol_selected in ROWCOL
            elif isinstance(rowcol_selected, tuple):
                """Facet-key '{rowcol_selected}' should have two strings
                '{row_or_col}' should be one of {LVLs_ROW} if other element is in COL-levels,
                or one of {LVLs_COL} if other element is in ROW-levels"""
                match_rc = ut.all_of_l1_in_l2(l1=rowcol_selected, l2=ROWCOL)
            else:
                match_rc = 0  # no match

            if match_rc:
                match = self._match_selected_xhue(
                    S=S, xhue_selected=xhue_selected, true_value=true_value
                )
                if match:
                    return match  # True
            else:
                match = 0

        return match

    def _match_include_exclude(
        self,
        PH: "pd.DataFrame" = None,
        exclude: list = None,
        include: list = None,
        exclude_in_facet: dict = None,
        include_in_facet: dict = None,
    ) -> "pd.DataFrame":
        """Matches Selection

        Returns:
            _type_: _description_
        """

        ### Defaults
        PH["incl."] = "All"  ##! INCLUDE ALL BY DEFAULT
        PH["incl. fac."] = 0
        PH["excl."] = 0
        PH["excl. fac."] = 0

        ### Match selection
        if not include is None:
            PH["incl."] = PH.apply(
                self._match_selected_xhue,
                xhue_selected=include,
                true_value="incl.",
                axis=1,
            )
        if not include_in_facet is None:
            PH["incl. fac."] = PH.apply(
                self._match_selected_rowcol,
                rowcol_selection_dict=include_in_facet,
                true_value="incl.",
                axis=1,
            )
        if not exclude is None:
            PH["excl."] = PH.apply(
                self._match_selected_xhue,
                xhue_selected=exclude,
                true_value="excl.",
                axis=1,
            )
        if not exclude_in_facet is None:
            PH["excl. fac."] = PH.apply(
                self._match_selected_rowcol,
                rowcol_selection_dict=exclude_in_facet,
                true_value="excl.",
                axis=1,
            )
        return PH

    @staticmethod
    def _conclude_include_exclude(S: pd.Series, exclude_over_include: bool) -> bool:
        """Concludes matches from selected exclusion/inclusion
        INCLUDE = NONE?
            >> INCLUDE EVERYTHING AND SPECIFY THINGS YOU WANT TO EXCLUDE (EXCLUDE OVER INCLUDE)
         INCLUDE = SPECIFIED?
            >> INCLUDE JUST THE DEFINED AND THEN EXCLUDE THINGS (EXCLUDE OVER INCLUDE)

        Args:
            S (pd.Series): _description_

        Returns:
            bool: _description_
        """
        incl = bool(S["incl."])
        incl_f = bool(S["incl. fac."])
        excl = bool(S["excl."])  ## NOT EXCLUDE MEANS INCLUDE
        excl_f = bool(S["excl. fac."])

        INCL = incl or incl_f
        EXCL = excl or excl_f

        if exclude_over_include:  # TODO: MIGHT BE BUGGY!
            SHOW = not EXCL if EXCL else INCL
        else:
            SHOW = INCL if INCL else not EXCL

        return SHOW

    #
    #
    # ... ANNOTATE POSTHOC  :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ... PostHoc: Main Annotation Function .............................................

    def iter__df_ax_ph(self, PH: "pd.DataFrame"):
        """Iterate through facet keys (row, col) and retrieve pieces of data, axes and posthoc

        Args:
            PH (_type_): _description_

        Yields:
            _type_: _description_
        """
        phG = PH.groupby(self.factors_rowcol)
        dfD = self.data_ensure_allgroups.groupby(self.factors_rowcol)
        axD = self.axes_dict

        ### Iterate through facet keys (row, col) and retrieve pieces of data, axes and posthoc
        for key in self.levelkeys_rowcol:
            print(key)
            ph = phG.get_group(key)
            df = dfD.get_group(key)
            ax = axD[key]
            yield df, ax, ph

    def _annotate_pairwise_base(self, PH: "pd.DataFrame"):
        for df, ax, ph in self.iter__df_ax_ph(PH):
            pass
            # ut.pp(df)
            # ut.pp(ph)
            # print(ax)

    def annotate_pairwise(
        self,
        only_sig: str = "strict",
        include: dict | list = None,
        exclude: dict | list = None,
        include_in_facet: dict = None,
        exclude_in_facet: dict = None,
        exclude_over_include=True,
        verbose=False,
    ):
        """Annotate pairs of groups with pairwise tests."""

        ### Assert presence of a posthoc table and plot
        # * Assert presence of Posthoc
        # * We could execute automatically, but producing a plot and a posthoc test at the same time is a lot to handle
        assert (
            not self.results.DF_posthoc is "NOT TESTED"
        ), "Posthoc not tested yet, please call .test_pairwise() first"
        PH = self.results.DF_posthoc

        ### Modify PH
        # * Reset Index for easy access
        PH.reset_index(inplace=True)

        # * Use only contrast rows from now on, if hue is present to ensure pairs have format (("l1", "l2"),("l3","l4"))
        if self.dims.hue:
            PH = PH.loc[PH["Contrast"].str.contains("*", regex=False), :]
        # ut.pp(PH)

        # ### Assert presence of Plot
        # assert (
        #     not self. is "NOT TESTED"
        # ), "Plot not tested yet, please call .test_pairwise() first"

        ### Check user argument selection: Go through assertions
        self._check_include_exclude(
            include=include,
            exclude=exclude,
            include_in_facet=include_in_facet,
            exclude_in_facet=exclude_in_facet,
        )

        ### Match user argument selection: Add columns 
        PH = self._match_include_exclude(
            PH=PH,
            include=include,
            exclude=exclude,
            include_in_facet=include_in_facet,
            exclude_in_facet=exclude_in_facet,
        )

        ### Conclude inclusion/exclusion
        PH["inc+exc"] = PH.apply(
            self._conclude_include_exclude,
            exclude_over_include=exclude_over_include,
            axis=1,
        )

        ### ... ANNOTATE
        self._annotate_pairwise_base(PH)

        ### Save PH
        self.results.DF_posthoc = PH

        ### Show PH if verbose
        if verbose:
            ut.pp(PH)

        return self


# %%

DF, dims = ut.load_dataset("fmri")
AN = Annotator(
    data=DF,
    dims=dims,
    subject="subject",
    verbose=True,
)

ph = AN.test_pairwise(paired=False, alpha=0.005)

# %%

AN = (
    AN.subplots()
    .fillaxes(kind="box")
    .annotate_pairwise(
        include=[0, "stim"],
        # exclude=[1, "cue", {1: ("cue", "stim")}],
        exclude=[1, {"stim": (0, 2)}],
        # exclude=[1, "cue", {"cue": ("cue", "stim")}], # ! Correct error
        # include_in_facet={"frontal": [0, "cue"], (0,1): [0, "cue"]}, # ! Correct error
        # include_in_facet={"frontal": [0, "cue"], "parietal": [0, "cue"]},
        # exclude_in_facet={"frontal": [2, "cue"], "parietal": [4, "stim"]},
        verbose=False,
    )
)


# %%
