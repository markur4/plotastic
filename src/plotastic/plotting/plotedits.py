#
# %% imports
from os import remove
from typing import TYPE_CHECKING, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt

# import pyperclip

# import markurutils as ut
import plotastic.utils.utils as ut
from plotastic.plotting.plottool import PlotTool

if TYPE_CHECKING:
    from plotastic.dataanalysis.dataanalysis import DataAnalysis


# %% Class PlotEdit:


class PlotEdits(PlotTool):
    #
    # == __init__ .......................................................................

    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)

        self._edit_y_scalechanged = False

    #
    # == EDIT .........................................................................

    #
    #' Shapes & Sizes ........................................#
    # TODO: Until now, stick with the arguments supplied to self.subplots

    #
    #' Titles of axes .........................................#

    @staticmethod
    def _standard_axtitle(
        key: tuple[str] | str,
        connect="\n",
        capitalize: bool = False,
    ) -> str:
        """Makes a standard title for axes

        :param key: _description_
        :type key: tuple[str] | str
        :param connect: _description_, defaults to "\n"
        :type connect: str, optional
        :param capitalize: _description_, defaults to False
        :type capitalize: bool, optional
        :return: _description_
        :rtype: str
        """        
        if isinstance(key, str):
            return ut.capitalize(key)
        elif isinstance(key, tuple):
            keys = []
            for k in key:
                if isinstance(k, str) and capitalize:
                    keys.append(ut.capitalize(k))
                else:
                    keys.append(str(k))  #' Can't capitalize int
            return connect.join(keys)

    def edit_titles(
        self,
        axtitles: dict = None,
    ) -> "PlotEdits | DataAnalysis":
        """Replaces Titles by axes keys

        :param axtitles: _description_, defaults to None
        :type axtitles: dict, optional
        :return: _description_
        :rtype: PlotEdits | DataAnalysis
        """        
        
        ### Assert correct input
        if not axtitles is None:
            keys = list(dict(self.axes_iter__keys_ax).keys())
            for key, ax in axtitles.items():
                assert key in keys, f"Can't edit titles, '{key}' should be one of {keys}"

        ### Edit titles
        if not axtitles is None:
            for key, ax in self.axes_iter__keys_ax:
                if key in axtitles.keys():
                    ax.set_title(axtitles[key])
        return self

    def edit_titles_with_func(
        self,
        row_func: Callable = None,
        col_func: Callable = None,
        connect="\n",
    ) -> "PlotEdits | DataAnalysis":
        """Applies formatting functions (e.g. lambda x: x.upper()) to
        row and col titles)

        :param row_func: _description_, defaults to None
        :type row_func: Callable, optional
        :param col_func: _description_, defaults to None
        :type col_func: Callable, optional
        :param connect: _description_, defaults to "\n"
        :type connect: str, optional
        :return: _description_
        :rtype: PlotEdits | DataAnalysis
        """        
        ### Default functions
        row_func = row_func or (lambda x: x)
        col_func = col_func or (lambda x: x)

        ### Don't use connect if only one facet
        # if self.factors_is_1_facet:
        #     connect=""
        
        for rowkey, axes in self.axes_iter__row_axes:
            for ax in axes:
                title = row_func(rowkey)
                ax.set_title(title)
        for colkey, axes in self.axes_iter__col_axes:
            for ax in axes:
                title = ax.get_title() + connect + col_func(colkey)
                ax.set_title(title)
        return self

    def edit_titles_replace(self, titles: list) -> "PlotEdits | DataAnalysis":
        """Edits axes titles. If list is longer than axes, the remaining
        titles are ignored

        :param titles: Titles to be set. The order of the titles
            should be the same as the order of the axes, which is from
            left to right for row after row (like reading)
        :type titles: list
        :return: _description_
        :rtype: PlotEdits | DataAnalysis
        """        
        

        for ax, title in zip(self.axes_flat, titles):
            if not title is None:
                ax.set_title(title)
        return self

    #
    #' Labels of x- & y-axis  .................................#

    def edit_xy_axis_labels(
        self,
        x: str = None,
        x_lowest_row: str = None,
        x_notlowest_row: str = None,
        y: str = None,
        y_leftmost_col: str = None,
        y_notleftmost_col: str = None,
    ) -> "PlotEdits | DataAnalysis":
        """Edits labels of x- & y-axis for all axes or for specific axes.

        Selection of specific axes overwrites general selection

        :param x: x-axis label for all axes :class:`df` , defaults to None
        :type x: str, optional
        :param x_lowest_row: _description_, defaults to None
        :type x_lowest_row: str, optional
        :param x_notlowest_row: _description_, defaults to None
        :type x_notlowest_row: str, optional
        :param y: _description_, defaults to None
        :type y: str, optional
        :param y_leftmost_col: _description_, defaults to None
        :type y_leftmost_col: str, optional
        :param y_notleftmost_col: _description_, defaults to None
        :type y_notleftmost_col: str, optional
        :return: DataAnalysis object
        :rtype: PlotEdits | DataAnalysis
        """

        ### y-axis labels
        if not y is None:
            for ax in self.axes_flat:
                ax.set_ylabel(y)
        if not y_leftmost_col is None:
            for ax in self.axes_iter_leftmost_col:
                ax.set_ylabel(y_leftmost_col)
        if not y_notleftmost_col is None:
            for ax in self.axes_iter_notleftmost_col:
                ax.set_ylabel(y_notleftmost_col)

        ### x-axis labels
        if not x is None:
            for ax in self.axes_flat:
                ax.set_xlabel(x)
        if not x_lowest_row is None:
            for ax in self.axes_iter_lowest_row:
                ax.set_xlabel(x_lowest_row)
        if not x_notlowest_row is None:
            for ax in self.axes_iter_notlowest_row:
                ax.set_xlabel(x_notlowest_row)
        return self

    #
    #' Scale of x- & y-axis  ..................................#

    def edit_y_scale_log(
        self, base=10, nonpositive="clip", subs=[2, 3, 4, 5, 6, 7, 8, 9]
    ) -> "PlotEdits | DataAnalysis":
        for ax in self.axes_flat:
            ax.set_yscale(
                value="log",  #' "symlog", "linear", "logit", ...
                base=base,  #' Base of the logarithm
                nonpositive=nonpositive,  #' "mask": masked as invalid, "clip": clipped to a very small positive number
                subs=subs,  #' Where to place subticks between major ticks ! not working
            )

            # ax.yaxis.sety_ticks()

        ### Set the scaled flag to True, to warn user that annotations should be called after NOT before
        self._edit_y_scalechanged = True

        return self

    def edit_x_scale_log(
        self, base=10, nonpositive="clip", subs=[2, 3, 4, 5, 6, 7, 8, 9]
    ) -> "PlotEdits | DataAnalysis":
        for ax in self.axes_flat:
            ax.set_xscale(
                value="log",  #' "symlog", "linear", "logit", ...
                base=base,  #' Base of the logarithm
                nonpositive=nonpositive,  #' "mask": masked as invalid, "clip": clipped to a very small positive number
                subs=subs,  #' Where to place subticks between major ticks ! not working
            )
        return self

    #
    #' Ticks and their Labels .................................#

    def edit_y_ticklabel_percentage(
        self, decimals_major: int = 0, decimals_minor: int = 0
    ) -> "PlotEdits | DataAnalysis":
        for ax in self.axes_flat:
            ax.yaxis.set_major_formatter(
                mpl.ticker.PercentFormatter(xmax=1, decimals=decimals_major)
            )
            # TODO: Not working
            ax.yaxis.set_minor_formatter(
                mpl.ticker.PercentFormatter(xmax=1, decimals=decimals_minor)
            )
        return self

    def edit_y_ticklabels_log_minor(self, subs: list = [2, 3, 5, 7]):
        """Displays minor ticklabels for log-scales. Only shows those ticks whose rounded mantissa (the digits from a float) is in subs

        Args:
            subs (list, optional): Mantissas (the digits from a float). Defaults to [2, 3, 5, 7].

        Returns:
            _type_: _description_
        """
        for ax in self.axes_flat:
            #' Set minor ticks, we need ScalarFormatter, others can't get casted into float
            ax.yaxis.set_minor_formatter(
                mpl.ticker.ScalarFormatter(useOffset=0, useMathText=False)
            )

            #' Iterate through labels
            for label in ax.yaxis.get_ticklabels(which="minor"):
                # ? How else to cast float from mpl.text.Text ???
                label_f = float(str(label).split(", ")[1])  #' Cast to float
                mantissa = int(
                    round(ut.mantissa_from_float(label_f))
                )  #' Calculate mantissa
                if not mantissa in subs:
                    label.set_visible(
                        False
                    )  #' Set those not in subs to invisible
        return self

    @staticmethod
    def _exchange_ticklabels(ax, labels: list | str) -> None:
        """Exchange ticklabels iterating through labels and ticks. Works with labels having different length than labels

        Args:
            ax (_type_): _description_
            labels (list | str): _description_
        """

        ### Retrieve xticks
        ticks = ax.get_xticks()

        ### If we want labels removed:
        labels = [""] * len(ticks) if labels == "" else labels

        ### Change lables independently of numbers of ticks and previous labels!
        old_labels = [item.get_text() for item in ax.get_xticklabels()]
        i = 0
        while i < len(labels) and i < len(old_labels):
            old_labels[i] = labels[i]
            i += 1

        # == Set new labels
        ax.set_xticklabels(labels=old_labels)

    def edit_x_ticklabels_exchange(
        self,
        labels: list | str = None,
        labels_lowest_row: list | str = None,
        labels_notlowest_row: list | str = None,
    ) -> "PlotEdits | DataAnalysis":
        """Changes text of x-ticklabels

        Args:
            labels (list, optional): _description_. Defaults to None.
            labels_lowest_row (list, optional): _description_. Defaults to None.
            labels_notlowest_row (list, optional): _description_. Defaults to None.

        Returns:
            PlotTool | DataAnalysis: _description_
        """

        # == EDIT
        ### Labels for all axes:
        if not labels is None:
            for ax in self.axes_flat:
                self._exchange_ticklabels(ax, labels)
        if not labels_lowest_row is None:
            for ax in self.axes_iter_lowest_row:
                self._exchange_ticklabels(ax, labels_lowest_row)
        if not labels_notlowest_row is None:
            for ax in self.axes_iter_notlowest_row:
                self._exchange_ticklabels(ax, labels_notlowest_row)

        return self

    def edit_x_ticklabels_rotate(
        self,
        rotation: int = 0,
        ha: str = None,
        va: str = None,
        pad: float = None,
        rotation_mode: str = "anchor",
        **set_kws: dict,
    ) -> "PlotEdits | DataAnalysis":
        """_summary_

        :param rotation: _description_, defaults to 0
        :type rotation: int, optional
        :param ha: _description_, defaults to None
        :type ha: str, optional
        :param va: _description_, defaults to None
        :type va: str, optional
        :param pad: _description_, defaults to None
        :type pad: float, optional
        :param rotation_mode: _description_, defaults to "anchor"
        :type rotation_mode: str, optional
        :return: _description_
        :rtype: PlotEdits | DataAnalysis
        """

        # == KWS
        ### Redirect kwargs, provide function defaults

        set_KWS = dict(
            rotation=rotation,
            ha=ha,
            va=va,
            rotation_mode=rotation_mode,
            pad=pad,
        )
        ### Remove None values so we can detect user defined values
        set_KWS = ut.remove_None_recursive(set_KWS)

        ### Change kwargs depending on selection
        if 20 < rotation < 89:
            set_KWS["ha"] = "right"
            set_KWS["va"] = "center"
            set_KWS["rotation_mode"] = "anchor"
            set_KWS["pad"] = 2.5

        ### Use user defined values if present
        set_KWS.update(set_kws)

        ### separate KWS into different dicts, since matplotlib has special needs
        ticklabel_KWS = {k: v for k, v in set_KWS.items() if not k in ["pad"]}
        params_KWS = {k: v for k, v in set_KWS.items() if k in ["pad"]}

        # == Rotate
        for ax in self.axes.flatten():
            obj = ax.get_xticklabels()  #' Retrieve ticks
            plt.setp(obj, **ticklabel_KWS)
            ax.tick_params(axis="x", **params_KWS)

        return self

    #
    #' Grid ...................................................#

    def edit_grid(self) -> "PlotEdits | DataAnalysis":
        for ax in self.axes_flat:
            ax.yaxis.grid(True, which="major", ls="-", linewidth=0.5, c="grey")
            ax.yaxis.grid(True, which="minor", ls="-", linewidth=0.2, c="grey")
            ax.xaxis.grid(True, which="major", ls="-", linewidth=0.3, c="grey")
        return self

    #
    #' Legend .................................................#

    @property
    def legend_handles_and_labels(self):
        if (
            self.factors_rowcol
        ):  #' If we have row and col factors, we need to get the legend from the first axes
            handles, labels = self.axes.flatten()[0].get_legend_handles_labels()
        else:
            handles, labels = self.axes.get_legend_handles_labels()
        ### Remove duplicate handles from repeated plot layers
        by_label = dict(zip(labels, handles))
        handles = by_label.values()
        labels = by_label.keys()
        labels = [ut.capitalize(l) for l in labels]
        return handles, labels

    def edit_legend(
        self,
        reset_legend: bool = False,
        title: str = None,
        handles: list = None,
        labels: list = None,
        loc: str = "center right",
        bbox_to_anchor: tuple = (1.15, 0.50),
        borderaxespad: float = 4,
        pad: float = None,
        frameon: bool = False,
        **kws,
    ) -> "PlotEdits | DataAnalysis":
        """Adds standard legend to figure"""
        ### Prevent legend duplication:
        if reset_legend:
            self.remove_legend()
            
        ### An Alias for borderaxespad
        if not pad is None and borderaxespad == 4:
            borderaxespad = pad

        KWS = dict(
            title=ut.capitalize(self.dims.hue) if title is None else title,
            handles=self.legend_handles_and_labels[0]
            if handles is None
            else handles,
            labels=self.legend_handles_and_labels[1]
            if labels is None
            else labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            borderaxespad=borderaxespad,
            frameon=frameon,
            # fontsize=10, # !! overrides entry from rcParams
        )
        KWS.update(**kws)

        # == Set legend
        self.fig.legend(**KWS)
        return self

    #
    #' fontsizes ..............................................#

    def edit_fontsizes(
        self, ticklabels=10, xylabels=10, axis_titles=10
    ) -> "PlotEdits | DataAnalysis":
        """Edits fontsizes in [pt]. Does not affect legent or suptitle

        Args:
            ticklabels (int, optional): _description_. Defaults to 9.
            xylabels (int, optional): _description_. Defaults to 10.
            axis_titles (int, optional): _description_. Defaults to 11.

        Returns:
            PlotTool: _description_
        """

        for ax in self.axes_flat:
            ax.tick_params(
                axis="y", which="major", labelsize=ticklabels
            )  #' Ticklabels
            ax.tick_params(axis="y", which="minor", labelsize=ticklabels - 1)
            ax.tick_params(axis="x", which="major", labelsize=ticklabels)
            ax.tick_params(axis="x", which="minor", labelsize=ticklabels - 1)
            ax.yaxis.get_label().set_fontsize(xylabels)  #' xy-axis labels
            ax.xaxis.get_label().set_fontsize(xylabels)
            ax.title.set_fontsize(axis_titles)  #' Title
        return self

    def edit_tight_layout(self) -> "PlotEdits | DataAnalysis":
        plt.tight_layout()
        return self

    #
