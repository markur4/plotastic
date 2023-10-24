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
    # == Properties of Users's Script ==================================================

    @staticmethod
    def prevent_overwrite(
        filename: "str | Path",
        parent: "Path" = None,
        ret_parent=False,
    ) -> "str | Path":
        """Returns a new filename that has a number at the end, if the filename already
        exists.
        - Checks filenames in path that are similar to filename
        - If there are similar filenames with an index behind them, it gets the largest
          index
        - Adds plus one to that index and puts it at the end of filenames
        
        :param filename: str
        :param parent:
        :return: str
        """
        ### Convert to string if path
        filename = str(filename) if isinstance(filename, Path) else filename
        parent = parent if not parent is None else Path.cwd()

        ### Get a list of filenames that might be overwritten
        files = ut.glob_searchfilename(path=parent, filename=filename, rettype="str")

        ### Define Pattern Rules:
        # * Between Start (^) and end ($) of line
        # * fname:  Match all characters non-greedy ( .*? )
        # * index: : 1 or 2 repetitions of single digit ( \d{1,2} )

        pattern = re.compile(r"^(?P<fname>.*?)_(?P<index>\d{1,2})$", flags=re.MULTILINE)
        matches: list[dict] = ut.re_matchgroups(pattern=pattern, string=files)
        ### Extract their indices
        indices: list[int] = [int(matchD["index"]) for matchD in matches]
        fnames: list[str] = [matchD["fname"] for matchD in matches]

        ### Add plus one to max index
        newindex = 0
        # print(filename, indices)
        if indices:
            filename = fnames[0]
            newindex = max(indices) + 1

        if ret_parent:
            return parent / Path(f"{filename}_{newindex}")
        else:
            return f"{filename}_{newindex}"

    # ==
    # == Time info =====================================================================

    @property
    def current_day(self) -> str:
        self._current_day = date.today().strftime("%Y%m%d")
        return self._current_day
