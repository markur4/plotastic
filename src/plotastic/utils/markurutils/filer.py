from __future__ import annotations
import re

# import os
from pathlib import Path
from datetime import date, datetime
from copy import copy

import markurutils as ut
# import plotastic.utils.utils as ut

# from markurutils import UTILS as ut


"""THe time is the current import time"""
_IMPORTTIME = datetime.now().strftime("%H%M%S")

"""RUNTIME CONFIGURATION"""
RC = dict(
    prefix="_",
    overwrite=False,
    use_daytime=True,
    use_subsubfolder=True,
)


def set_option(
    prefix: str = None,
    overwrite: bool = None,
    use_daytime: bool = None,
    use_subsubfolder: bool = None,
    cwd: str | Path = None,
):
    """


    :param overwrite: bool, (optional)
        | If True, every output files will be overwritten with each script execution
        | If False, Filenames are automatically adapted to become unique and prevent overwriting
        | defaults to False
    :param use_subsubfolder: bool, str, (optional)
        | If False: Will save results within a subfolder of current working environment.
        | - e.g.  ./subfolder/result.pdf
        | - e.g.  ./qPCR-Analysis/GAPDH_boxplot.pdf
        | If True, will make another subfolder within that subfolder of current working environment.
        | - e.g.  ./subfolder/sub-subfolder/result.pdf
        | - e.g.  ./qPCR-Analysis/GAPDH/GAPDH_boxplot.pdf
        | defaults to True
    :param use_daytime: str, (optional)
        | If True: adds time to end of filename:
        | - The subfolder will get a date (YYYYMMDD), the sub-subfolder will get time (HHMMSS)
        | - e.g.  ./subfolder_YYYYMMDD/sub-subfolder_HHMMSS/result.pdf
        | - e.g.  ./qPCR-Analysis_20221231/GAPDH_235959/boxplot.pdf
        | defautls to True
    :param prefix: str, (optional)
        | A prefix added before every folder and filename.
        | - e.g.  ./Prefix_subfolder_YYYYMMDD/Prefix_sub-subfolder_HHMMSS/result.pdf
        | - e.g.  ./Prefix_qPCR-Analysis_20221231/Prefix_GAPDH_235959/GAPDH_boxplot.pdf
        | If no sub-subfolder:
        | - e.g.  ./Prefix_qPCR-Analysis_20221231/Prefix_GAPDH_boxplot_235959.pdf
        | defaults to "_"
    :param cwd: str or pathlib.Path, redirect the current working directory for

    :Example:

    >>> ### Basic usage
    >>> f = ut.Filer(title="GAPDH", use_daytime=False) # Specify title of the whole project.
    >>> f.filename # Title will be part of the base filename, can be modified
    'GAPDH'
    >>> f.path_file_parent # Output filepath uses two levels of directories: named by Script, and named by title
    PosixPath('./qPCR/_qPCR/_GAPDH')
    >>> f.path_file # Base Filename, prefixes and suffixes can be added
    PosixPath('./qPCR/_qPCR/_GAPDH/GAPDH')

    >>> ### Switch on daytime suffixes (by default True)
    >>> f = ut.Filer(title="GAPDH", use_daytime=True)
    >>> f.filename
    'GAPDH'
    >>> f.path_file_parent # The folder to store all results
    PosixPath('./qPCR/_qPCR_20220820/_GAPDH_125047')
    >>> f.path_file # Base Filename, prefixes and suffixes can be added
    PosixPath('./qPCR/_qPCR_20220820/_GAPDH_125047/GAPDH')
    >>> f.path_file

    >>> ### Add your own prefix ("_" by default)
    >>> f = ut.Filer(title="GAPDH", use_daytime=False, prefix="#hello#_")
    >>> f.path_file
    PosixPath('./qPCR/#hello#_qPCR/#hello#_GAPDH/GAPDH')

    >>> ### Not using a sub-subfolder will save file into
    >>> f = ut.Filer(title="GAPDH", use_daytime=False, prefix="#hello#_", use_subsubfolder=False)
    >>> f.path_file
    PosixPath('./qPCR/#hello#_qPCR/GAPDH')

    >>> ### Files will never be overwritten, unless you say so (by default True)
    >>> f = ut.Filer(title="GAPDH", use_daytime=False, prefix="#hello#_", use_subsubfolder=False, overwrite=False)
    >>> f.path_file # suppose GAPDH is already present, _1 will be added to end
    PosixPath('./qPCR/#hello#_qPCR/GAPDH_1')

    """
    global RC
    oldRC = RC.copy()
    if not prefix is None:
        RC["prefix"] = prefix
    if not overwrite is None:
        RC["overwrite"] = overwrite
    if not use_subsubfolder is None:
        RC["use_subsubfolder"] = use_subsubfolder
    if not use_daytime is None:
        RC["use_daytime"] = use_daytime

    if not cwd is None:
        WorkingDirectory.cwd = Path(cwd).resolve()

    """Print differences to defaults"""
    print(f"#! CONFIGURING FILER: Switching these switches to: {set(oldRC) ^ set(RC)}")


class WorkingDirectory(ut.OOP):
    """
    Outpath Hierarchy:
    - O_CWD =
        - Current Working Directory
        - Folder where all the results are stored
        - By default, it's the folder where the script is. Can be set to another location
    - O_CWD_DAY      = Inside the current Working Directory, each day another folder will be made, in order to prevent overwriting of previous analyses
    - O_CWD_DAY_TIME = Inside
    """

    """Script name, path, filepath"""
    SCRIPT_NAME: str = ut.get_scriptname()  # The filename
    SCRIPT_PATH: "Path" = Path().cwd()
    SCRIPT_EXTENSION = ".ipynb" if ut.is_notebook() else ".py"
    SCRIPT_FILEPATH: "Path" = (SCRIPT_PATH / SCRIPT_NAME).with_suffix(SCRIPT_EXTENSION)
    cwd: "Path" = SCRIPT_PATH

    def __init__(self):
        self.current_time = _IMPORTTIME
        self._current_day = date.today().strftime("%Y%m%d")

        self._is_notebook = None

    ### SET CWD (CURRENT WORKING DIRECTORY)'''
    @classmethod
    def set_cwd(cls, path: "Path") -> "Path":
        cls.cwd = Path(path)
        return cls.cwd

    # @property
    # def current_time(self) -> str:
    #     self._current_time = datetime.now().strftime("%H%M%S")
    #     return self._current_time

    @property
    def is_notebook(self) -> bool:
        self._is_notebook = ut.is_notebook()
        return self._is_notebook

    @property
    def current_day(self) -> str:
        self._current_day = date.today().strftime("%Y%m%d")
        return self._current_day


class Filer(
    WorkingDirectory,  # FilerSettings
):
    """
    An assistant for Analysis Objects that provides path for outup files
    Folder Hierarchy of Output :
    - Results are always saved in a subfolder within the current working directory
    - Within that subfolder, you can choose if you want to save results in a sub-subfolder
     (e.g. ./subfolder/subsubfolder/result.pdf).
    Rationale:
    Suppose you want to redo an analysis, improve it. You do not want to save the output in the same folder as in the
    last analysis. So you make a new subfolder within the current working environment ( ./subfolder2).
    It makes sense to mark that subfolder with the current date (YYMMDD), since this analysis is unique for today.
    During your analysis you will execute the analysis several times. Your subfolder will be cluttered, since you do not
    want to overwrite your results with each execution because you want to have the possibility to compare new results
    with previous ones.
    One way to organize the results of this analysis is to cluster all the different result from one execution into
    another sub-subfolder (./subfolder2/sub-subfolder). In order to prevent overwriting, Each subfolder gets a unique
     name, e.g. the time of execution (HHMMSS).
    This way one can execute scripts without never having to worry about overwriting/losing results
    But In the end, one needs to manually delete all those different analyses.

    """

    def __init__(self, title="untitled"):
        """An Assistant that generates output paths starting at the current working environment
        You can change working environment using :class:`ut.set_cwd(path=yourpath)`

        | Is part of :class:`ut.Analysis` class (but uses :class:`pyrectories.Filer` if :class:`pyrectories` is
        installed, extending its functionality with a :class:`pyrectories.Finder` class).

        :param title: str, (optional)
            Title of the project. By default, it will be part of the :class:`ut.Filer.filename`
            defaults to "untitled""


        :Example:

        >>> ### ut.Filer constructs its paths using cwd and the name of the executing script
        >>> import uttils as ut
        >>> ut.get_scriptname() # Prints stem of filename of the executing script/notebook
        'qPCR'
        >>> from pathlib import Path
        >>> Path.cwd()  # Print current working directory
        PosixPath('./qPCR')
        """

        ### Get self.finder and self.CWD'''
        # WorkingDirectory.__init__(self)
        super().__init__()

        ### Settings'''
        # FilerSettings.__init__(self)
        ### Paths'''
        self._path_subfolder = None  # * -> /cwd/DATE_scriptname/result_1.pdf
        self._path_subsubfolder = (
            None  # * -> /cwd/DATE_scriptname/TIME_title_suffix/result_1.pdf
        )
        self._parent = None
        self._path_file = None
        ### Filename'''
        self.title = title
        # self._filename_current = None
        self._filename = None

    ### Subfolder & Sub-Subfolders #.......................................................................................................

    @property
    def path_subfolder(self) -> "Path":
        if RC["use_daytime"]:
            foldername = f"{RC['prefix']}{self.SCRIPT_NAME}_{self.current_day}"
        else:
            foldername = f"{RC['prefix']}{self.SCRIPT_NAME}"
        self._path_subfolder = self.cwd / Path(foldername)
        return self._path_subfolder

    @property
    def path_subsubfolder(self) -> "Path":
        if RC["use_daytime"]:
            # foldername = f"{self.PREFIX}{self.title}_{self.current_time}"
            foldername = f"{RC['prefix']}_{self.current_time}"
        else:
            foldername = f"{RC['prefix']}{self.title}"
        self._path_subsubfolder = self.cwd / self.path_subfolder / Path(foldername)
        return self._path_subsubfolder

    @property
    def parent(self) -> "Path":  # keep it as a ge
        if RC["use_subsubfolder"]:
            parent = self.path_subsubfolder
        else:
            parent = self.path_subfolder
        self._parent = parent
        return self._parent

    ### Subfolder & Sub-Subfolders #.......................................................................................................

    def add_to_title(self, to_end: str = "", to_start: str = "", con: str = "_") -> str:
        """
        :param to_start: str, optional (default="")
        String to add to start of title
        :param to_end: str, optional (default="")
        String to add to end of title
        :param con: str, optional (default="_")
        Conjunction-character to put between string addition and original title
        :return: str
        """
        self.title = f"{to_start}{con}{self.title}"
        self.title = f"{self.title}{con}{to_end}"
        return self.title

    @staticmethod
    def prevent_overwrite(
        filename: str | "Path", parent: "Path" = None, ret_filepath=False
    ) -> str | "Path":
        """
        - Checks filenames in path that are similar to filename
        - If there are similar filenames with an index behind them, it gets the largest index
        - Adds plus one to that index and puts it at the end of filenames
        :type path: pathlib.Path
        :param parent:
        :param filename: str
        :return: str
        """

        filename = str(filename) if isinstance(filename, Path) else filename
        # parent = parent  if parent  else self.parent #self.get_parent()

        """# Get a list of filenames that might be overwritten"""
        files = ut.glob_searchfilename(path=parent, filename=filename, rettype="str")

        """Define Pattern Rules:"""
        """
        - Between Start (^) and end ($) of line
        - fname:  Match all characters non-greedy ( .*? )
        - index: : 1 or 2 repetitions of single digit ( \d{1,2} )
        """
        pattern = re.compile(r"^(?P<fname>.*?)_(?P<index>\d{1,2})$", flags=re.MULTILINE)
        matches: list[dict] = ut.re_matchgroups(pattern=pattern, string=files)
        """# Extract their indices """
        indices: list[int] = [int(matchD["index"]) for matchD in matches]
        fnames: list[str] = [matchD["fname"] for matchD in matches]

        """# Add plus one to max index"""
        newindex = 0
        # print(filename, indices)
        if indices:
            filename = fnames[0]
            newindex = max(indices) + 1

        if ret_filepath:
            return parent / Path(f"{filename}_{newindex}")
        else:
            return f"{filename}_{newindex}"

    @property
    def filename(self):
        """Generates a filename from given :class:`ut.Filer.title`"""
        self._filename = self.get_filename()
        return self._filename

    ### INTERFACE FUNCTIONS: Make directories & Construct names"""
    def make_parentdir(self) -> "Path":
        """Returns and makes a parent directory for outputs."""

        """Make folder"""
        parent = self.parent  # self.get_parent()
        if not parent.is_dir():
            print(f"#! Making directory: {parent}")
            # os.makedirs(parent)
            parent.mkdir(parents=True, exist_ok=True)
        return parent

    def get_filename_base(self, titlesuffix: str = None):
        """filename without overwrite protection"""
        title = copy(self.title)
        titlesuffix = f"_{titlesuffix}" if titlesuffix else ""

        return f"{title}{titlesuffix}"

    def get_filename(
        self,
        titlesuffix: str = None,
        overwrite=None,
    ) -> "Path":
        """
        :param titlesuffix: str, It will extend the title of the project by e.g. plot-kind
        :param extension: str, The file extension like .pdf, .png, etc.
        :return:
        """

        ### Core of the filename
        title = copy(self.title)
        titlesuffix = f"_{titlesuffix}" if titlesuffix else ""

        DEFAULT = f"{title}{titlesuffix}"
        # default = self.prevent_overwrite(BASE)

        ### Object-Level Interface: OVERWRITE, DAYTIME, SUBSUBFOLDER
        path_file = self.parent / Path(
            title
        )  # using property results in recursive loop
        if RC["overwrite"] or overwrite:
            N = DEFAULT
            if path_file.is_file():
                print("#! FILE EXISTING! O V E R W R I T I N G !!")
        else:
            ### Check if file is present
            if RC["use_daytime"]:  # * Add Timeindex if set
                if RC["use_subsubfolder"]:
                    N = self.prevent_overwrite(DEFAULT, parent=self.parent)
                else:  # * Add Time
                    _n = f"{title}{titlesuffix}_{self.current_time}"
                    N = self.prevent_overwrite(_n, parent=self.parent)

            ### Add Index to prevent overwriting
            elif path_file.is_file():
                print("#! FILE EXISTING, not overwriting")
                N = self.prevent_overwrite(DEFAULT, parent=self.parent)

            else:
                print("#! File not existing yet")
                N = DEFAULT

        self._filename = Path(N)
        return self._filename

    def insert_folder(self, filepath: str | "Path") -> "Path":
        """
        Takes a filepath and inserts a subfolder before the filename (for e.g. multiple exports of one dataset per timepoint.
        :param prefix: str. Will be added in front of the new subfolder
        :param filepath: filepath ('/user/experiment/bla1')
        :param undo_overwrite_protection:
        :return:         filepath ('/user/experiment/bla1/bla1')
        """

        ### CAST PATH and REMOVE FILE EXTENSION'''
        filepath = Path(filepath)

        filename = filepath.name
        filestem = (
            self.get_filename_base()
        )  # equals filepath.stem. need this to remove the integeger at the end from overwrite protection
        parent = Path(filepath).with_suffix("").parent
        # print(filepath)
        ### Update Overwrite protection'''
        filename = self.prevent_overwrite(filename=filename, parent=parent)

        ### CONSTRUCT NEW FILEPATH'''
        subfolder = parent / filestem
        subfolder.mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(subfolder):
        #     os.mkdir(subfolder)
        filepath_new = parent / filestem / filename
        return filepath_new

    # ==Outpath of any file ..................................................................................................
    def get_filepath(
        self,
        titlesuffix: str = None,
        makeparent=True,
        add_folder=False,
        overwrite=None,
    ) -> (
        "Path"
    ):  ### Outpath of any file #####################################################
        overwrite = overwrite if not overwrite is None else RC["overwrite"]
        name = self.get_filename(titlesuffix=titlesuffix, overwrite=overwrite)
        filepath = self.parent / name

        if add_folder:
            filepath = self.insert_folder(filepath)
            pass  # TODO: This feature does not work

        ###Make folder
        if makeparent:
            self.make_parentdir()

        return filepath


# ? DEPRECATED: make_outs...........................................................'''
# def make_outs(location=".", globalname=None, outsuffix="time", subfolder=True, overwrite=False):
#     """In the same folder as the jupyternotebook we want a folder with the "globalname" + "titlesuffix" as foldername. In that folder there are all plots.
#     The filenames of these plots are appended with the current time, so they don't get overwritten

#     :param location: The path where to place the out folder. Standard is set to same location as the jupyter notebook
#     :param globalname: How to call the outfolder folders. Standard is the same filename as jupyterlab (REQUIRES !pip install ipynbname)
#     :param outsuffix: Its what added to the out-folders name. Standard is today's date, so that subsequent anylses of the same data don't overwrite old analyses
#     :param subfolder: Choose whether to use subfolders in the outfolders
#     :return:
#     """

#     from datetime import date, datetime
#     import os
#     import ipynbname

#     # from uttils.inout import export

#     ### Define the Prefix of the main Out-Folder.
#     if outsuffix == "time":
#         current_date = date.today().strftime("%Y%m%d")
#         outsuffix = f"_{current_date}"

#     ### The outfoldername should contain the globalname of this experiment
#     if not globalname:
#         if ut.is_notebook():
#             globalname = ipynbname.name()
#         else:
#             globalname = Path(__file__).name
#         print(f"### make_outs: Global File Name is: \n  {globalname}")

#     ### Construct Outputfolderpath
#     absloc = os.path.abspath(location)
#     outFolderName = globalname + outsuffix
#     outFolderPath = os.path.join(absloc, outFolderName)
#     if not os.path.exists(outFolderPath):
#         os.mkdir(outFolderPath)

#     ### Generate Out variable with absolute path and FIRST HALF of the file's name
#     current_time = datetime.now().strftime("%H%M%S")
#     if overwrite:
#         Out = os.path.join(outFolderPath, f"{globalname}_")
#     else:
#         Out = os.path.join(outFolderPath, f"{globalname}_{current_time}_")

#     ### Construct Output Subfolderpath
#     if subfolder:
#         current_time = datetime.now().strftime("%H%M%S")
#         if overwrite:
#             outSubFolderName = "overwrite"
#         else:
#             outSubFolderName = current_time
#         outSubFolderPath = os.path.join(outFolderPath, outSubFolderName)
#         if os.path.exists(outSubFolderPath)==False:
#             os.mkdir(outSubFolderPath)

#         ### Make the Filename
#         Out = os.path.join(outSubFolderPath, f"{globalname}_")

#     return Out
