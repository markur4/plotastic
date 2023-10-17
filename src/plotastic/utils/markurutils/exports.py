from __future__ import annotations
from typing import Collection, Sequence

import os
from pathlib import Path
import ipynbname

from joblib import Memory
import tempfile

import pandas as pd

import weasyprint

from markurutils import UTILS as ut


"""Cache up e.g. converted ipynb notebooks (if unchanged)"""
MEMORY_TEMP = Memory(tempfile.gettempdir(), verbose=0)


"""# IPYNB............................................................................................"""


def ipynb_to_docx(notebook: str = None, outpath=".") -> str:
    notebook = notebook if notebook else ipynbname.name() + ".ipynb"
    # name = ipynbname.name()
    name = Path(notebook).stem
    out = f"{os.path.join(outpath, name)}"
    c = f"pandoc '{notebook}' -s -o '{out}'.docx  # CONVERT INTO DOCX!"
    print(c)
    os.system(c)

    return out


@MEMORY_TEMP.cache
def ipynb_to_py(notebook: str | "Path" = None, lastmodified=None) -> None:
    _ = lastmodified  # acquired by `Path.lstat().st_mtime`, required for joblib cache to keep updated
    notebook = notebook if notebook else ipynbname.name() + ".ipynb"
    out = Path(notebook).stem  # !! nbconvert will use same directory as source file!!
    c = f"jupyter nbconvert '{notebook}' --to python --output '{out}'"
    print(c)
    os.system(c)


def ipynbs_to_pys(notebooks: Collection[str]) -> list:
    """Takes List of ipynb paths and covnerts 'em all to .py files"""
    pys = []
    for nb in notebooks:
        nbpath = Path(nb).absolute()
        lastmodified = nbpath.lstat().st_mtime  # time of last modification
        ipynb_to_py(notebook=nbpath, lastmodified=lastmodified)

        nbname = Path(nb).stem
        pys.append(Path(nb).parent / (nbname + ".py"))
    return pys


"""# Pandas ......................................................................................"""


def export_df(
    df,
    out: "Path" = Path("./dataframe"),
    subfolder=True,
    formats: Sequence = ("xslx", "html", "pdf", "latex"),
    stylefunc=None,
    printLatex=False,
    float_format="{:.3e}".format,
) -> None:
    """
    Exports table to .xlsx, Latex and html

    :param df: pandas dataframe
    :param out: Output filepath
    :param formats: ("xslx", "html", "pdf", "latex")
    :param printLatex:
    :param stylefunc:
    :param subfolder:
    :param float_format:
    :return:
    """

    """Check formats"""
    default_formats = ("xlsx", "html", "pdf", "latex")
    formats = (formats,) if type(formats) is str else formats
    for f in formats:
        assert f in default_formats, f"#! {f} should have been one of {default_formats}"

    """# Put all formats into subfolder"""
    if subfolder:
        # out = ut.add_subfolder(out)
        out = ut.insert_subfolder(filepath=out)

    """### Define Standard style for every table"""

    # with pd.option_context('display.float_format', float_format):
    def standard_style(styler):
        """defines standard"""
        cells = {
            "selector": "td",
            "props": """font-family: Arial; font-size: 11px""",
        }
        index_names = {
            "selector": ".index_name",
            "props": "font-style: italic; font-size: 12px; color: grey; font-weight:normal;",
        }
        headers = {
            "selector": "th:not(.index_name)",
            "props": """font-family: Arial; font-size: 12px; text-align: center""",
        }
        styler.set_table_styles([cells, index_names, headers])
        # formatter = "{:.2%}" #convert to percent
        # formatter = {('Decision Tree', 'Tumour'): "{:.2f}",
        #              ('Regression', 'Non-Tumour'): lambda x: "$ {:,.1f}".format(x*-1e6)}
        # styler.format(precision=2, thousands=" ", formatter=formatter,
        # na_rep='MISSING'
        # )
        # styler.background_gradient(axis=None, vmin=-0.03, vmax=0.03, cmap="seismic")

        ### Apply Custom Styler functions
        if stylefunc:
            styler = stylefunc(styler)
        return styler

    dfs = df.style.pipe(standard_style)

    """# CONVERT"""
    if "xlsx" in formats:
        dfs.to_excel(out.with_suffix(".xlsx"))

    """Change format of floats"""
    float_cols = [c for c in df.dtypes.index if "float" in str(df.dtypes[c])]
    dfs = dfs.format(dict(zip(float_cols, [lambda x: float_format(x)])))
    if "html" in formats:
        html = dfs.to_html()
        with open(out.with_suffix(".html"), "w") as handle:
            handle.write(html)

    if "pdf" in formats:
        """pdf is made by converting existing html into a pdf"""
        html = dfs.to_html()
        with open(out.with_suffix(".html"), "w") as handle:
            handle.write(html)
        weasyprint.HTML(out.with_suffix(".html")).write_pdf(out.with_suffix(".pdf"))
        if not "html" in formats:
            os.remove(out.with_suffix(".html"))

    if "latex" in formats:
        latex = df.to_latex(float_format=float_format)
        with open(str(out) + "_latex.txt", "w") as handle:
            handle.write(latex)
    if printLatex:
        latex = df.to_latex()
        print("\n", "# ## Printing Latex:\n\n", latex)
