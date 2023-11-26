#
# %% imports

from typing import Callable

import plotastic.utils.utils as ut

# %% Test wrapping function


if __name__ == "__main__":
    descr = """
    Mode of overwrite protection. If "day", it simply adds the current date at the end
    of the filename, causing every output on the same day to overwrite itself. If
    "nothing" ["day", "nothing"], files with the same filename will be detected in the
    current work directory and a number will be added to the filename. If True,
    everything will be overwritten.
    """
    w = ut.wrap_text(descr)
    print(w)
    len("            ")


# %% Write :param: part of docstring


def param(
    param: str,
    descr: str,
    default: str = "",
    typ: str = "",
    optional: bool = False,
) -> str:
    """Returns part of docstring describing parameter in sphinx format"""

    ### If descr starts with new line remove it
    if descr.startswith("\n"):
        descr = descr[1:]

    S = []

    ### First line, (no tabstop needed)
    # # Don't include :param: in docstring, add that manually always, so
    # # vscode at least shows the parameter in the intellisense
    S.append(" ")  #' whitespace after :param param:
    # S = f":param {param}: {wrap_descr(descr)}"
    S.append(
        ut.wrap_text(
            string=descr,
            width=72,
            width_first_line=54,
            indent="            ",
        )
    )

    ### Add default value to first line
    if default:
        if isinstance(default, str):
            # # Add quotes if param defaults to string
            default = f"'{default}'"
        S.append(f", defaults to {default}")

    ### Further options need a tab
    ### Type
    if typ:
        S.append("\n\t")  #' newline
        S.append(f":type {param}: {typ}")

    ### Optional, same line as type
    if optional:
        S.append(f", optional")

    return "".join(S)


if __name__ == "__main__":
    docpart = param(
        param="sdaf",
        descr="makes makes and does does stuffystuff",
        default="ja!",
        typ="str",
    )
    print(docpart)


# %% Substitute variables in docstring


def subst(*args, **kwargs):
    """Decorator that substitutes variables in docstrings, e.g.: {} as args and {var} as
    kwargs
    """

    def F(func: Callable):
        doc = func.__doc__
        ### Shouldn't raise error if no docstring is present
        if doc:
            try:
                ### Substitute args
                func.__doc__ = doc.format(*args, **kwargs)
            except KeyError as e:
                raise KeyError(
                    f"Could not substitute {e} in docstring of {func.__name__}"
                    "with {args} or {list(kwargs.keys())}"
                )

        return func

    return F


if __name__ == "__main__":
    # p = """:param verbose: Set to False to not print stuff, defaults to False"""
    # p += "\n\t:type verbose: bool"
    p = param(
        param="verbose",
        descr="Ladidah awesome parameter if you know what I mean. Makes makes and does does stuffystuff",
        default="ja!",
        typ="str",
    )

    @subst("banana", var2="milkshake", var3=p)
    def bla(verbose: False):
        """this is a docstring with {} and {var2},

        :param verbose: {var3}
        """
        if verbose:
            print("jo!")

    print(bla.__doc__)


# %%

### Overwrite Protection
param_overwrite = param(
    param="overwrite",
    descr="""
    Mode of overwrite protection. If "day", it simply adds the current date at the end
    of the filename, causing every output on the same day to overwrite itself. If
    "nothing" ["day", "nothing"], files with the same filename will be detected in the
    current work directory and a number will be added to the filename. If True,
    everything will be overwritten.
    """,
    default="day",
    typ="str | bool",
    optional=True,
)

if __name__ == "__main__":
    from plotastic.dataanalysis.dataanalysis import DataAnalysis

    # print(overwrite)
    print(DataAnalysis.save_statistics.__doc__)
    # DataAnalysis.save_fig()

# %%
