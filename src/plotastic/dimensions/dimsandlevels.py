# !!
# %% Imports


from collections import defaultdict

# %% Imports
# from operator import index, le
from re import L  # for type hinting my Class type for return values
from typing import Dict, Callable, TYPE_CHECKING

from copy import copy
from itertools import product

import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch

# from scipy.stats import skew as skewness
import matplotlib.pyplot as plt
import seaborn as sns


# import markurutils as ut
import plotastic.utils.utils as ut

from plotastic.dimensions.dims import Dims

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

# %% Utils
df = None  #' Prevent warning when using catchstate


def catchstate(df, var_name: str = "df"):
    ## THIS FUNCTIONS DOESN'T WORK WHEN IMPORTED FROM MODULE
    """
    Helper function that captures intermediate Dataframes.
    In the global namespace, make a new variable called var_name and set it to dataframe
    :param df: Pandas dataframe
    :param var_name:
    :return:
    """
    globals()[var_name] = df

    return df


# %% Class DimsAndLevels =======================================================


class DimsAndLevels:
    """Handles dimensions and levels within imported dataframe"""

    def __str__(self):
        # d = self.__dict__
        D = {
            a: getattr(self, a)
            for a in dir(self)
            if (
                not a.startswith("_")
                and not callable(getattr(self, a))
                # and not isinstance(getattr(self, a), ut.Filer)
            )
        }

        ### Catch unprintable types
        if type(D.get("data")) is pd.DataFrame:
            D["data"] = (D["data"].shape, list(D["data"].columns))
            D["data_ensure_allgroups"] = (
                D["data_ensure_allgroups"].shape,
                list(D["data_ensure_allgroups"].columns),
            )
        if "fig2nd" in D:
            D["fig2nd"] = f"{len(D['fig2nd'])} axes"
        # if "ax" in D:
        #     D["ax"] = len(self.ax)

        return ut.printable_dict(D=D, start_message=f"{type(self)}: ")

    # ==
    # == INIT ==================================================================

    def __init__(
        self,
        data: pd.DataFrame,
        dims: dict | Dims,
    ):
        """Handles dimensions and levels within dataframe

        :param data: Pandas dataframe, long-format!
        :type data: pd.DataFrame
        :param dims: Dims object storing x, y, hue, col, row.
        :type dims: dict | Dims
        :raises AssertionError: If dims contains entries that are not in columns
        """

        ### Save data and dims
        self.data: pd.DataFrame = data
        self.dims: Dims = dims if type(dims) is Dims else Dims(**dims)

        ### Make sure dims are present in data
        self._assert_dims_with_data(df=data, dims=self.dims)

    @staticmethod
    def _assert_dims_with_data(df: pd.DataFrame, dims: Dims) -> None:
        """Asserts that all entries in dims are present in data

        :param df: DataFrame to be checked
        :type df: pd.DataFrame
        :param dims: Dims object storing x, y, hue, col, row.
        :type dims: Dims
        :raises AssertionError: If dims contains entries that are not in columns
        :return: None
        """

        for dim in dims.asdict(incl_None=False).values():
            assert (
                dim in df.columns
            ), f"#! '{dim}' not in columns, expected one of {df.columns.to_list()}"

    # ==
    # ==
    # == List FACTORS ==========================================================

    @property  #' [row, col, hue, x] (dims may be missing)
    def factors_all(self) -> list[str]:
        F = (self.dims.row, self.dims.col, self.dims.x, self.dims.hue)
        return [e for e in F if (not e is None)]

    @property  #' [row, col, hue]
    def factors_all_without_x(self) -> list[str]:
        """Used for iterating through dataframe yielding lists of groups"""
        F = (self.dims.row, self.dims.col, self.dims.hue)
        return [e for e in F if (not e is None)]

    # @property
    # def factors_all_include_none(self) -> list[str]:
    #     F = (self.dims.row, self.dims.col, self.dims.hue, self.dims.x)
    #     return [e for e in F]

    @property  #' {"y": dims.y, "x": dims.x, "hue": dims.hue, "col": dims.col, "row": dims.row}
    def factors_as_dict(self) -> dict:
        """
        gets the dimensions in forms of a dictinary to be passed onto seaborn functions
        :return:
        {"y": self.dims.y, "x": self.dims.x,"hue": self.dims.hue, "col": self.dims.col, "row": self.dims.row}
        :rtype: dict
        """
        return self.dims.asdict(incl_None=False)
        # return {dim: factor for dim, factor in self.dims.asdict().items() if not factor is None}

    @property  #' [col1, col7, col8]
    def columns_not_factor(self) -> list[str]:
        return [c for c in self.data.columns if c not in self.factors_all]

    @property
    def factors_xhue(self) -> str | list[str]:
        if self.dims.hue:
            xhue = [self.dims.x, self.dims.hue]
        else:
            xhue = self.dims.x
        return xhue
    
    
    @property  #' [row, col]; [row]; [col]; [""]
    def factors_xhue_list(self) -> list[str]:
        if self.dims.hue:
            xhue = [self.dims.x, self.dims.hue]
        else:
            xhue = [self.dims.x]
        return xhue
    
    
    # @property
    # def factors_huex(self) -> str | tuple[str]:
    #     """
    #     When within/between/factors has two elements, it DOES matter in which order you post them.
    #     Usually, multiple t-tests don't always compare two groups coming from some combination of factors and levels, since they assume that these combinations don't make sense, and sometimes they really don't.
    #     We still want to be able to get every combination and do this by simply switching the order of factors.
    #     This allows us to compare e.g. not just every 'treatment type' per 'timepoint', but also every 'timepoint' per 'treatment type'.
    #     """
    #     if self.dims.hue:
    #         xhue = (self.dims.hue, self.dims.x)
    #     else:
    #         xhue = self.dims.x
    #     return xhue

    # @property
    # def factors_xhue_huex(self) -> tuple[str] | tuple[tuple[str]]:
    #     """Makes sure that x is not present twice in the list if no hue is present"""
    #     return tuple(set(self.factors_xhue, self.factors_huex))

    # @property
    # def factors_xhue_list(self) -> list[str]:
    #     if self.dims.hue:
    #         xhue = [self.dims.x, self.dims.hue]
    #     else:
    #         xhue = [self.dims.x]
    #     return xhue

    @property  #' [row, col]; row; col; None
    def factors_rowcol(self) -> str | list[str] | None:
        if self.dims.row and self.dims.col:
            rowcol = [self.dims.row, self.dims.col]
        elif self.dims.row:
            rowcol = self.dims.row
        elif self.dims.col:
            rowcol = self.dims.col
        else:
            rowcol = None
        return rowcol

    @property  #' [row, col]; [row]; [col]; [""]
    def factors_rowcol_list(self) -> list[str]:
        if self.dims.row and self.dims.col:
            rowcol = [self.dims.row, self.dims.col]
        elif self.dims.row:
            rowcol = [self.dims.row]
        elif self.dims.col:
            rowcol = [self.dims.col]
        else:
            rowcol = [""]
        return rowcol

    # == Properties of FACTORS  ================================================

    @property  #' no hue, row or col
    def factors_is_just_x(self) -> bool:
        return not self.dims.row and not self.dims.col and not self.dims.hue

    @property  #' Either just row or col.
    def factors_is_1_facet(self) -> bool:
        only_row = (not self.dims.row is None) and (self.dims.col is None)
        only_col = (self.dims.row is None) and (not self.dims.col is None)
        return only_row or only_col

    @property  #' No col or row
    def factors_is_unfacetted(self) -> bool:
        return not self.dims.row and not self.dims.col

    @property  #' {"f1": "continuous", "f2": "category",}
    def factors_types(self) -> dict:
        """Returns: {"f1": "continuous", "f2": "category",}"""
        D = dict()
        for factor in self.factors_all:
            type = self.data[factor].dtype.name
            if type == "object":
                D[factor] = "object"
                print(
                    f"#! factor '{factor}' is of type object so it's probably a string"
                )
            if type == "category":
                D[factor] = "category"
            elif type in [
                "int",
                "float",
                "float32",
                "float64",
                "int32",
                "int16",
            ]:
                D[factor] = "continuous"
            else:
                print(f"#!!! factor '{factor}' is of unknown type '{type}'")
                D[factor] = "unknown"
        return D

    @property
    def factors_categoric(self):
        """Includes only columns that were defined as nominal or ordinal"""
        raise NotImplementedError

    #
    # == Retrieve FACTORS ======================================================

    #' input: Hue -> "smoke"
    def getfactors_from_dim(
        self, putative_factors: str | list[str,]
    ) -> str | list[str]:
        """Get column name, if "x" or "hue" is passed instead of actual column name"""
        if isinstance(putative_factors, str):
            putative_factors = [putative_factors]

        result = []
        dimsD = self.dims.asdict()
        for f in putative_factors:
            if f in self.data:
                result.append(f)
            elif f in dimsD:
                factor = dimsD[f]
                # assert not factor is None, f"#! Can't set {putative_factors} as factors, since '{f}' is assigned to None.\n\tUse one of {self.dims.asdict(incl_None=False, incl_by=False)} "
                result.append(factor)
            else:
                raise AssertionError(
                    f"#!! '{f}' should have been one of these dimensions or column names: \n\t{list(dimsD.keys() )}\n\t{list(self.data.columns)}"
                )

        if len(result) == 1:
            return result[0]
        else:
            return result

    def get_factor_from_level(self, level: str, ret_notfound=None):
        """Gets the factor from a level"""
        for factor, levels in self.levels_dict_factor.items():
            if level in levels:
                return factor
        ### If nothing returned, nothing was wound
        if ret_notfound is None:
            return None  #' We use this to check if the level is in the data
        elif ret_notfound == "raise":
            raise AssertionError(f"#! Level '{level}' not found in data.")

    # def get_rank_from_level(self, level: str):
    #     """Gets the factor from a level"""
    #     for rank, levels in self.hierarchy.items():
    #         if level in levels:
    #             return rank

    #
    # == LEVELS ================================================================

    def get_levels_from_column(self, colname: str) -> list[str]:
        """Returns: [lvl1, lvl2]"""
        S = self.data[colname]
        # if S.dtype.name == "category":
        if isinstance(S.dtype, pd.api.types.CategoricalDtype):
            return S.cat.categories.to_list()
        else:
            return S.unique().tolist()

    @property  #' {"f1": [lvl1, lvl2], "f2": [lvl1, lvl2],}
    def levels_dict_factor(self) -> dict:
        """Returns: {"f1": [lvl1, lvl2], "f2": [lvl1, lvl2],}"""
        return {
            factor: self.get_levels_from_column(colname=factor)
            for factor in self.factors_all
        }

    @property  #' {"row":[row_l1, row_l2, ...], "col":[c_l1, c_l2, ...], "hue":[...], "x":[...]}
    def levels_dict_dim(self) -> dict:
        """Returns: {"row":[row_l1, row_l2, ...], "col":[c_l1, c_l2, ...], "hue":[...], "x":[...]}"""
        D = self.levels_dict_factor
        return {
            "row": D.get(self.dims.row, ""),
            "col": D.get(self.dims.col, ""),
            "hue": D.get(self.dims.hue, ""),
            "x": D.get(self.dims.x),
        }

    @property  #' [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2), (hue_lvl1, hue_lvl2), (x_lvl1, x_lvl2)]
    def levels_tuples(self) -> list[tuple]:
        """Returns: [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2), (x_lvl1,
        x_lvl2), (hue_lvl1, hue_lvl2)]"""
        return [
            tuple(l) for l in self.levels_dict_factor.values() if not l is None
        ]

    @property  #' [(row, R_lvl1), (row, R_lvl2), (col, C_lvl1), (col, C_lvl2), ...]
    def levels_factortuples(self) -> list[tuple]:
        """Make a list of tuples with first element being factor and second element
        being level
        :return: [(row, R_lvl1), (row, R_lvl2), (col, C_lvl1), (col, C_lvl2), ...]
        """
        return [
            (factor, level)
            for factor, levels in self.levels_dict_factor.items()
            for level in levels
        ]

    @property  #' [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2) ]
    def levels_tuples_rowcol(self):
        """Returns: [(R_lvl1, R_lvl2), (C_lvl1, C_lvl2) ]"""
        return [
            tuple(l)
            for k, l in self.levels_dict_factor.items()
            if (not l is None) and (k in ut.ensure_list(self.factors_rowcol))
        ]

    @property  #' [ (R_l1, C_l1, X_l1, Hue_l1), (R_l1, C_l2, X_l1, Hue_l1), (R_l2, C_l1, X_l1, Hue_l1), ... ]
    def levelkeys_all(
        self,
    ) -> list[tuple]:  # !! refactored from 'levelkeys' -> 'levelkeys_all'
        """Contains ALL possible combinations of levels, even if they don't exist in the data.

        :return: [ (R_l1, C_l1, X_l1, Hue_l1), (R_l1, C_l2, X_l1, Hue_l1), (R_l2,
        C_l1, X_l1, Hue_l1), ... ]."""
        return [key for key in product(*self.levels_tuples)]

    @property  #' [ (R_l1, C_l1), (R_l1, C_l2), (R_l2, C_l1), ... ]
    def levelkeys(self) -> list[tuple]:
        """Contains unique combinations of levels that exist in the data. Should be
        sorted (?)

        :return: [ (R_l1, C_l1, X_l1, Hue_l1), (R_l1, C_l2, X_l1, Hue_l1), (R_l2, C_l1, X_l1, Hue_l1), ... ].
        :rtype: _type_
        """

        return list(self.data.groupby(self.factors_all).groups.keys())

    @property  #' [ (R_l1, C_l1), (R_l1, C_l2), (R_l2, C_l1), ... ]
    def levelkeys_rowcol(self) -> list[tuple | str]:
        """Returns: [ (R_l1, C_l1), (R_l1, C_l2), (R_l2, C_l1), ... ]"""
        return [
            key
            if not len(key) == 1
            else key[0]  #' If only one factor, string is needed, not a tuple
            for key in product(*self.levels_tuples_rowcol)
        ]

    @property  #' (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
    def levels_xhue_flat(self) -> tuple:
        """Returns: (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)"""
        l = []
        for factor in (self.dims.x, self.dims.hue):
            if factor is None:
                continue
            S = self.data[factor]
            if S.dtype.name == "category":
                [l.append(e) for e in S.cat.categories.to_list()]
            else:
                [l.append(e) for e in S.unique()]
        return tuple(l)

    @property  #' (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)
    def levels_rowcol_flat(self) -> tuple:
        """Returns: (x_lvl1, x_lvl2, x_lvl3, hue_lvl1, hue_lvl2)"""
        l = []
        for factor in (self.dims.row, self.dims.col):
            if factor is None:
                continue
            S = self.data[factor]
            if S.dtype.name == "category":
                [l.append(e) for e in S.cat.categories.to_list()]
            else:
                [l.append(e) for e in S.unique()]
        return tuple(l)

    # ==
    # == Properties of Levels ==================================================

    @property
    def len_rowlevels(self) -> int:
        if not self.dims.row is None:
            return len(self.levels_dict_factor[self.dims.row])
        else:
            return 1  #' Used by subplots, we need minimum of one row

    @property
    def len_collevels(self) -> int:
        if not self.dims.col is None:
            return len(self.levels_dict_factor[self.dims.col])
        else:
            return 1  #' sed by subplots, we need minimum of one col

    # ==
    # == COUNT FULLY CONNECTED LEVELS ==========================================

    def _count_levelcombos(self) -> defaultdict:
        """For each level of each factor count how often it appears together with another
        level of every other factor

        :return: Dictionary with levelpairs as keys and appearance count as values
        :rtype: defaultdict (doesn't require initialisation of keys
        """
        ### Create a dictionary to store the counts of level combinations
        level_combocount = defaultdict(int)

        ### Iterate through the list of level keys
        #' To initialize, we need all possible combinations of levels from all factors
        for level_key in self.levelkeys_all:
            ### Iterate through all pairs of levels in the level key
            for i in range(len(level_key)):
                for j in range(i + 1, len(level_key)):
                    ### Sort the levels in alphabetical order to count combinations regardless of order
                    # !! can't sort tuples with different types
                    # combination = tuple(sorted((level_key[i], level_key[j])))
                    combination = tuple(
                        set((level_key[i], level_key[j]))
                    )  #' WORKING
                    level_combocount[combination] += 1

        return level_combocount

    def levels_combocounts(self, normalize=False, heatmap=True) -> pd.DataFrame:
        """Makes a DataFrame or its heatmap comparing every level with each other, counting how often they
        appear together in the data. This is useful to see how well the factors structure
        the data. If a factor has levels that are always found together, the levels have the
        largest number in that DataFrame, meaning that's factor is a good candidate to facet
        the data by.

        :param normalize: Normalizes result by max value, defaults to False
        :type normalize: bool
        :param heatmap: If True, returns a heatmap, otherwise a DataFrame, defaults to True
        :type heatmap: bool
        :return: DatFrame with all levels as columns and rows, and the count of how
            often they appear together as values
        :rtype: pd.DataFrame
        """

        ### Get level_combocount
        #' dict with every pairwise combination of levels from every factor as keys and
        #' count as values
        levelcombo_count: dict = self._count_levelcombos()

        ### Initialize the DataFrame with multiindexes
        index = pd.MultiIndex.from_tuples(
            self.levels_factortuples, names=["factor", "level"]
        )
        df = pd.DataFrame(0, columns=index, index=index)

        ### Update the DataFrame with the counts from level_combinations
        for levelkey, count in levelcombo_count.items():
            f0 = self.get_factor_from_level(level=levelkey[0])
            f1 = self.get_factor_from_level(level=levelkey[1])
            df.loc[(f0, levelkey[0]), (f1, levelkey[1])] = count
            df.loc[
                (f1, levelkey[1]), (f0, levelkey[0])
            ] = count  #' Ensure symmetry

        ### Normalize the DataFrame by dividing by the maximum value
        #' make percent format, round to 0 decimals
        if normalize:
            df = df / df.max().max() * 100
            df = df.round(0).astype(int)

        ### Plot Heatmap, way more helpful than DataFrame
        if heatmap:

            def heatmap_combocount(matrix: np.array, depth: int):
                ### Colors:
                custom_cmap = ut.make_cmap_saturation(
                    undersat=(0.5, 0.0, 0.0),  #' dark red
                    oversat=(0.0, 0.7, 0.0),  #' arker green
                    n=depth,
                )

                ### Create a heatmap of the inconsistency matrix
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(matrix, cbar=True, cmap=custom_cmap)
                s = "Counts how often each level appears with another in the same data point"
                s += "\nGreen = levels appear together in ALL data points"
                s += "\nGrey = levels appear together in part of data points"
                s += "\nRed =  levels appear together in 0 data points"
                plt.title(s, fontsize=10)
                plt.xlabel("Levels")
                plt.ylabel("Levels")
                return fig

            fig = heatmap_combocount(matrix=df, depth=df.max().max())

        return df

    # ==
    # == Dendrogramm ===========================================================

    @staticmethod
    def _jaccard_similarity(
        combination1: list | tuple,
        combination2: list | tuple,
    ) -> float:
        """Calculates the Jaccard similarity:
        - We take two combinations:
        - e.g. ("A", 2, "C", "D")
        - and  ("A", 1, "C", "D"),
        - Measure the length of the intersection (=1)
        - Measure the length of the union (=7)
        - Divide the intersection by the union (1/7 = 0.14)

        :param combination1: Some listlike object
        :type combination1: list | tuple
        :param combination2: Some listlike object
        :type combination2: list | tuple
        :return: Similarity between two lists
        :rtype: float
        """
        ### Take unique values,
        #' Yes the Score might change, but relations between scores won't change
        set1 = set(combination1)
        set2 = set(combination2)

        ### Calculate Jaccard similarity
        len_intersection = len(set1.intersection(set2))
        len_union = len(set1) + len(set2) - len_intersection
        return len_intersection / len_union

    @staticmethod
    def _link_levelkeys(
        levelkeys: list[tuple[str]], method: str = "ward"
    ) -> np.array:
        """Calculates Jaccard distance matrix between levelkeys and links them by
        hierarchical clustering.
        :param method: How to link distance matrix by sch.linkage.
            ["ward","single","complete","average"], defaults to "ward"
        :param levelkeys: List of tuples with levels from each factor. E.g. [(R_lvl1,
            C_lvl1, X_lvl1, Hue_lvl1), (R_lvl2, C_lvl1, X_lvl2, Hue_lvl1)]. Can be
            obtained with DF.set_index(Factors).unique()), defaults to None
        :return: Linkage matrix
        :rtype: np.array
        """
        ### Take Unique values from the index
        #' The number of occurences of each index is not important
        #' They often represent technical replicates
        #' It's more important how often the levels occur together within one element
        ### levelkeys are the same as index after setting DF.index to all factors
        # !! keep this static, since it's useful.
        # levelkeys = self.levelkeys if levelkeys is None else levelkeys

        ### Create a square distance matrix based on Jaccard similarity
        n = len(levelkeys)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = 1 - DimsAndLevels._jaccard_similarity(
                    levelkeys[i], levelkeys[j]
                )

        ### Find links for hierarchical clustering
        Z: np.array = sch.linkage(dist_matrix, method=method)

        return Z

    @staticmethod
    def _plot_dendrogram_from_levelkeys(
        levelkeys: list[tuple[str]],
        clustering_method: str = "ward",
        ret: bool = False,
        **dendrogramm_kws,
    ):
        """Plots a dendrogramm that shows the hierarchical clustering of each levelkeys.
        It helps determining how the data is organized by each factor's levels.

        :param levelkeys: List of tuples with levels from each factor. E.g. [(R_lvl1,
            C_lvl1, X_lvl1, Hue_lvl1), (R_lvl2, C_lvl1, X_lvl2, Hue_lvl1), ...]. Can be
            obtained with DF.set_index(Factors).index.unique()), defaults to None
        :type levelkeys: list[tuple[str]]
        :param clustering_method: How to link distance matrix by sch.linkage.
            ["ward","single","complete","average"], defaults to "ward"
        :type clustering_method: str, optional
        :param ret: _description_, defaults to False
        :type ret: bool, optional
        :return: A dictionary with dendrogramm info and a matplotlib plot
        :rtype: dict
        """

        ### Get linkage matrix
        Z = DimsAndLevels._link_levelkeys(
            levelkeys=levelkeys,
            method=clustering_method,
        )

        ### Initialize figure
        plt.figure(figsize=(3, (len(Z) / 10) + 2))  # ? (w, h) Works good!

        ### Plot Dendrogramm
        dendro: dict = sch.dendrogram(
            Z,
            labels=levelkeys,
            # labels=self.levelkeys_all, #* Same as DF.index.unique() if factors are set as index
            orientation="right",
            **dendrogramm_kws,
        )

        ### Plot edits
        # plt.legend() # ? not working
        plt.title("Combination Hierarchy")
        plt.xlabel("Linkage Distance")

        if ret:
            return dendro

    def levels_dendrogram(
        self,
        clustering_method="ward",
        ret=False,
        **dendrogramm_kws,
    ) -> dict:
        """Plots a dendrogramm that shows the hierarchical clustering of each levelkeys.
        It helps determining how the data is organized by each factor's levels.

        :param clustering_method: How to link distance matrix by sch.linkage.
            ["ward","single","complete","average"], defaults to "ward"
        :type clustering_method: str, optional
        :param ret: Return dendrogram, defaults to False
        :type ret: bool, optional
        :return: A dictionary with dendrogramm info and a matplotlib plot
        :rtype: dict
        """

        ### If no facets this gives cryptic error, redirect that
        assert (
            not self.factors_is_just_x
        ), "#! Can't plot dendrogramm for X only, need at least one facet (hue, row, col)."

        ### Plot dendrogram
        dendro = self._plot_dendrogram_from_levelkeys(
            levelkeys=self.levelkeys,
            clustering_method=clustering_method,
            ret=ret,
            **dendrogramm_kws,
        )

        if ret:
            return dendro

    # ==
    # == SETTERS ===============================================================

    # !!
    def switch(
        self,
        *keys: str,
        inplace=False,
        verbose=True,
        **kwarg: str | Dict[str, str],
    ) -> "DimsAndLevels | DataAnalysis":
        a = self if inplace else copy(self)

        #' NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply
        a.dims = a.dims.switch(*keys, inplace=inplace, verbose=verbose, **kwarg)

        return a

    # !!
    def set(
        self,
        dims: "Dims | dict" = None,
        y: str = None,
        x: str = None,
        hue: str = None,
        row: str = None,
        col: str = None,
        data: "pd.DataFrame" = None,
        transform: str | Callable = None,
        title: str = None,  # type: ignore
        inplace=False,
        verbose=True,
    ) -> "DimsAndLevels | DataAnalysis":
        """Redefines values of Analysis.dims (y,x,hue,row,col) and also title,"""

        ### HANDLE COPY"""
        a = self if inplace else ut.copy_by_pickling(self)

        if not dims is None:
            if isinstance(dims, dict):
                a.dims = Dims(**dims)
            else:
                a.dims = dims

        if any((y, x, hue, row, col)):
            kwargs = {
                key: arg
                for key, arg in zip(
                    ("y", "x", "hue", "row", "col"), (y, x, hue, row, col)
                )
                if not arg is None
                #' WE ALSO ALLOW "none" TO REMOVE A DIMENSION
            }

            #' NEEDS RESETTING, otherwise in-chain modifications with inplace=False won't apply"""
            a.dims = a.dims.set(inplace=inplace, verbose=verbose, **kwargs)
            # a.dims.set(inplace=inplace, verbose=verbose, **kwargs) # NOT WORKIN IN-CHAIN
        if not data is None:
            a.data = data
            # print(data)
        if not transform is None:
            a.y_transform(func=transform)
        if not title is None:
            a.title = title
        return a

    # def update_dims(
    #     self,
    #     y=None,
    #     x=None,
    #     hue=None,
    #     row=None,
    #     col=None,
    #     data: "pd.DataFrame" = None,
    #     # transform: str | Callable = None,
    #     title: str = None,
    #     inplace=False,
    #     verbose=True,
    # ) -> "DimsAndLevels | DataAnalysis":
    #     """Same as `self.update_analysis`, just with inplace=True"""

    #     return self.set(
    #         y=y,
    #         x=x,
    #         hue=hue,
    #         row=row,
    #         col=col,
    #         data=data,
    #         title=title,
    #         inplace=inplace,
    #         verbose=verbose,
    #     )


# !!
# !!
# !!    #* #####################################################################


# %%

# DF, dims = plst.load_dataset("tips")
# dims = dict(y="tip", x="sex", hue="day", col="smoker", row="time")

# DA = Analysis(
#     data=DF,
#     dims=dims,
#     # subject="day",
#     verbose=True,
# )


# %% automatic testing


# def tester(DF, dims):
#     A = DimsAndLevels(data=DF, dims=dims, verbose=True)  # .switch("x", "col")


dimses = [
    dict(y="tip", x="day", hue="sex", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="smoker", row="time"),
    dict(y="tip", x="sex", hue="day", col="time", row="smoker"),
    dict(y="tip", x="sex", hue="day", col="time"),
    dict(y="tip", x="sex", hue="day", row="time"),
    dict(y="tip", x="sex", hue="day", row="size-cut"),
    dict(y="tip", x="sex", hue="day"),
    dict(y="tip", x="sex"),
    dict(y="tip", x="size-cut"),
]

# DF, dims = plst.load_dataset("tips")
# for dim in dimses:
#     print("\n !!!", dim)
#     tester(DF, dim)


# %% Prototypes
# !! not working
"""! I tried implementing generators that iterate through the data vy looping through the list of
levelkeys and then picking out the datawindow using grouped.get_group(key). 
This is cool since I can make a function that uses skip=True/False as an argument, but generators don't accept arguments that easily
Also, I don't think I need it
"""
# def skip():
#     grouped = DA.data.groupby(DA.factors_all)
#     for key in DA.levelkeys_all:
#         try:
#             df = grouped.get_group(key)
#         except KeyError:
#             continue
#         yield key, df


# def notskip():
#     grouped = DA.data.groupby(DA.factors_all)
#     for key in DA.levelkeys_all:
#         try:
#             df = grouped.get_group(key)
#         except KeyError:
#             #' Associate key with respective factor
#             factor_and_keys = {DA.get_factor_from_level(lvl): lvl for lvl in key}
#             factor_and_keys_df = pd.DataFrame(factor_and_keys, index=[0])

#             empty_df = pd.DataFrame(columns=DA.data.columns)

#             #' Append to empty df
#             df = empty_df.append(factor_and_keys_df, ignore_index=True)

#         yield key, df


# def data_iter__key_allgroups(skip_empty_groups=False):
#     if skip_empty_groups:
#         return skip
#     else:
#         return notskip


# # for key, df in data_iter__key_allgroups(skip_empty_groups=True):
# #     print(key, df)
# #     print()

# for key, df in skip():
#     print(key, df)
#     print()

# for key, df in notskip():
#     print(key, df)
#     print()
