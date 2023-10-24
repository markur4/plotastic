#
# %% Imports

import sys

import ipynbname
import IPython

import inspect

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

    @property
    def _current_shell(self) -> str:
        """Returns the name of the current shell executing this script.
        :raises: NameError if not in IPython
        """
        return IPython.get_ipython().__class__.__name__

    @property
    def _script_is_notebook(self) -> bool:
        """checks from where the script is executed. Handy if you want to print()"""

        ### .py file run in VS code interactive mode..?
        # ! not working,
        # variables: dict = IPython.extract_module_locals()[1]
        # if "__vsc_ipynb_file__" in variables:
        #     self._is_notebook = False
        #     return self._is_notebook

        ### Other Possibilities
        try:
            shell = self._current_shell
            if shell == "ZMQInteractiveShell":
                self._is_notebook = True  #* Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                self._is_notebook = False  # * Terminal running IPython
            else:
                self._is_notebook = False  # * Other type (?)
        except NameError:
            self._is_notebook = False  # * Probably standard Python interpreter

        return self._is_notebook

    @property
    def _script_name(self) -> str:
        """Generates a filename from user's script

        :return: Filename of the user's script
        :rtype: str
        """
        frame = inspect.currentframe()
        while frame.f_back:
            frame = frame.f_back
        filename = frame.f_globals['__file__']
        return filename
        
        
        
        # current_script = __file__ # self.DEFAULT_TITLE
        # exec("current_script = Path(__file__).stem")
        # exec("print(Path(__file__).stem)")
        # return current_script        
        # exec("return Path(__file__).stem")
        

        # if self._is_notebook:
        #     ### Jupyter Notebook or similar
        #     try:
        #         ### Does not work in interactive mode (e.g. in VSCode)
        #         scriptname = ipynbname.name()
        #     except FileNotFoundError:
        #         scriptname = self.DEFAULT_TITLE
        #         # ! not working
        #         # variables: dict = IPython.extract_module_locals()[1]
        #         # ### Vscode
        #         # scriptname = variables.get("__vsc_ipynb_file__", self.DEFAULT_TITLE)
        # else:
        #     ### Standard .py file run in terminal (?)
        #     scriptname = Path(sys.argv[0]).stem
        #     # os.path.basename(sys.argv[0]) #Path().name
        # return scriptname

    # ==
    # == Time info =====================================================================

    @property
    def current_day(self) -> str:
        self._current_day = date.today().strftime("%Y%m%d")
        return self._current_day
