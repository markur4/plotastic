#
# %% Imports

# import sys

# import ipynbname
# import IPython

# import inspect

import re

import markurutils as ut

# from IPython import get_ipython

from datetime import date

from pathlib import Path
from typing import Any


# %% Class Filer
class Filer:

    """A class to handle file operations.
    - It reads the name of the current file, and sets it as the default filename for
      saving.
    - Provides funcrtion for overwrite protection.
    - More stuff coming..?


    """

    DEFAULT_TITLE = "plotastic_result"

    # ==
    # == __init__ ======================================================================

    def __init__(self, title: str):
        self.title = title

    # ==
    # == Time info =====================================================================

    @property
    def current_day(self) -> str:
        return date.today().strftime("%y%m%d")

    # ==
    # == Properties of Users's Script ==================================================

    @staticmethod
    def _prevent_overwrite_all(filename: str) -> str:
        """Returns a new filename that has a number at the end, if the filename already
        exists.
        - Checks filenames in path that are similar to filename
        - If there are similar filenames with an index behind them, it gets the largest
          index
        - Adds plus one to that index and puts it at the end of filenames

        :param filename: filename.
        :type filename: str
        :return: str
        """

        ### Get a list of filenames that might be overwritten
        files = ut.glob_searchfilename(
            path=Path.cwd(),
            filename=filename,
            rettype="str",
        )

        ### Define Pattern Rules:
        # * Between Start (^) and end ($) of line
        # -- fname:  Match all characters non-greedy ( .*? )
        # ! fname: Match exact string
        # * index: : 1 through 3 repetitions of single digit ( \d{1,3} )
        # regex = r"^(?P<fname>.*?)_(?P<index>\d{1,2})$" # ? old one
        regex = r"^(?P<fname>" + filename + r")_(?P<index>\d{1,3})$"

        ### Get matches
        pattern = re.compile(regex, flags=re.MULTILINE)
        matches: list[dict] = ut.re_matchgroups(pattern=pattern, string=files)
        ### Extract their indices
        indices: list[int] = [int(matchD["index"]) for matchD in matches]
        ### fnames are never used
        # fnames: list[str] = [matchD["fname"] for matchD in matches]

        ### Add plus one to max index
        newindex = 0
        if indices:
            newindex = max(indices) + 1

        return f"{filename}_{newindex}"

    def prevent_overwrite(self, filename: "str | Path", mode: str = "day") -> str:
        """Returns a new filename that has a number or current date at the end to enable
        different modes of overwriting protection.

        :param filename: filename to be protected from overwriting
        :type filename: str | Path
        :param mode: Mode of overwrite protection. If "day", it simply adds the
            current date at the end of the filename, causing every output on the same
            day to overwrite itself. If "nothing" ["day", "nothing"], files with the
            same filename will be detected in the current work directory and a number
            will be added to the filename, defaults to "day"
        :type mode: str, optional
        :return: filename that is protected from overwriting by adding either number or
            the current date at its end
        :rtype: str
        """
        mode_args = ["day", "nothing"]
        assert mode in mode_args, f"mode must be one of {mode_args}, not {mode}"

        ### Convert to string if path
        filename = str(filename) if isinstance(filename, Path) else filename
        ### Remove suffix
        filename = filename.split(".")[0]

        if mode == mode_args[0]:  # * "day"
            filename = f"{filename}_{self.current_day}"
        elif mode == mode_args[1]:  # * "nothing"
            filename = self._prevent_overwrite_all(filename=filename)

        return filename
