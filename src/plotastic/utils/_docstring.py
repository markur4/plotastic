#
# %% imports
from typing import Callable


# %% Write :param: part of docstring


def param(
    param: str,
    descr: str,
    default: str = "",
    typ: str = "",
    **kwargs,
):
    """Returns part of docstring describing parameter in sphinx format"""

    ### First line, (no tabstop needed)
    S = f":param {param}: {descr}"
    
    ### Add default value to first line
    if default:
        if isinstance(default, str):  # * Add quotes
            default = f"'{default}'"
        S += f", defaults to {default}"

    ### Further options need a tab
    if typ:
        S += f"\n\t:type {param}: {typ}"

    # for keyword, argument in kwargs.items():
    #     pass
    # match keyword:
    #     case "default":
    #     case "type":
    return S


if __name__ == "__main__":
    docpart = param(
        param="sdaf",
        descr="makes makes and does does stuffystuff",
        default="ja!",
        typ="str",
    )
    print(docpart)


# %%
def subst(*args, **kwargs):
    """Decorator that substitutes {variables} in docstrings"""

    def F(func: Callable):
        doc = func.__doc__
        ### Shouldn't raise error if no docstring is present
        if doc:
            func.__doc__ = doc.format(*args, **kwargs)

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
