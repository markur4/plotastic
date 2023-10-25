#
# %% imports

from typing import Callable


# %% A function that wraps a multiline string into 88 characters


def wrap_descr(string: str, width: int = 76, width_first_line: int = 58) -> str:
    """Wraps a multiline string into a certain width. If first_line is specified, it
    will remove those characters from the first line before wrapping.

    :param string: A multiline string
    :type string: str
    :param width: Width of characters to wrap to, defaults to 72
    :type width: int, optional
    :param width_first_line: Width of characters to remove from first line, defaults to
        18
    :type width_first_line: int, optional
    :return: Wrapped string
    :rtype: str

    """
    ### Return if string is already short enough
    if len(string) <= width:
        return string

    ### Remove any newline and tabs
    string = string.replace("\n", " ").replace("\t", " ")

    ### Split into words
    words = string.split(" ")

    ### Remove empty words and strip whitespace
    words = [w.strip() for w in words if w.strip()]

    ### Wrap first line. It's 18 chars shorter to make room for :param param:
    text = ""
    while len(text + words[0]) < width_first_line:
        text += words.pop(0) + " "
    text = text[:-1]  # Remove last space
    text += "\n"

    ### Wrap remaining lines
    lines = []
    line = "            "  # * 12 spaces
    while words:
        if len(line + words[0]) <= width:
            line += words.pop(0) + " "
        else:
            lines.append(line.rstrip())
            line = ""
    if line:
        lines.append(line.rstrip())

    ### join lines, add 12 spaces as indent
    text += "\n            ".join(lines)

    return text


if __name__ == "__main__":
    descr = """
    Mode of overwrite protection. If "day", it simply adds the current date at the end
    of the filename, causing every output on the same day to overwrite itself. If
    "nothing" ["day", "nothing"], files with the same filename will be detected in the
    current work directory and a number will be added to the filename. If True,
    everything will be overwritten.
    """
    w = wrap_descr(descr)
    print(w)
    len("            ")


# %%

# %% Write :param: part of docstring


def param(
    param: str,
    descr: str,
    default: str = "",
    typ: str = "",
    optional: bool = False,
):
    """Returns part of docstring describing parameter in sphinx format"""

    ### If descr starts with new line remove it
    if descr.startswith("\n"):
        descr = descr[1:]

    ### First line, (no tabstop needed)
    S = f":param {param}: {wrap_descr(descr)}"

    ### Add default value to first line
    if default:
        if isinstance(default, str):  # * Add quotes
            default = f"'{default}'"
        S += f", defaults to {default}"

    ### Further options need a tab
    ### Type
    if typ:
        S += f"\n\t:type {param}: {typ}"

    ### Optional, same line as type
    if optional:
        S += f", optional"

    return S


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
                    f"Could not substitute {e} in docstring of {func.__name__} with {args} or {list(kwargs.keys())}"
                )

        return func

    return F


if __name__ == "__main__":
    p = """:param verbose: Set to False to not print stuff, defaults to False"""
    p += "\n\t:type verbose: bool"
    p = param(
        param="return_stuff",
        descr="Ladidah awesome parameter if you know what I mean. Makes makes and does does stuffystuff",
        default="ja!",
        typ="str",
    )

    @subst("banana", var2="milkshake", var3=p)
    def bla(verbose: False):
        """this is a docstring with {} and {var2},

        {var3}
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
    print(DataAnalysis.save_fig.__doc__)
    # DataAnalysis.save_fig()
