from os import dup
from typing import Hashable, TYPE_CHECKING  # , List, Optional, Set, Tuple, Union
import collections
import re

# if TYPE_CHECKING:
import pandas as pd
import numpy as np


# <editor-fold desc="''' 2 Builtin Datatypes ">

import collections.abc

from decimal import Decimal

### FLOAT #.......................................................................................................


def exponent_from_float(number: float):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def mantissa_from_float(number: float):
    return Decimal(number).scaleb(-exponent_from_float(number)).normalize()


### DICTIONARIES #.......................................................................................................
def get_key(q, dic):
    """

    :param val:
    :param dic: {k1: [1,2,3,"et"], k2: [4,"wsef"]}
    :return: key if q is in val
    """
    result = "not found"
    for k, v in dic.items():
        if q in v:
            result = k

    return result


def update_dict_recursive(d: dict, u: dict) -> dict:
    """Recursively update a dict.

    Args:
        d (dict): Dictionary to be updated
        u (dict): Dictionary to update with

    Returns:
        dict: New dictionary
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def remove_None_recursive(d: dict) -> dict:
    """Recursively remove None values from a dict.

    Args:
        d (dict): Dictionary to be updated

    Returns:
        dict: New dictionary
    """

    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, collections.abc.Mapping):
            d[k] = remove_None_recursive(v)
        elif v is None:
            del d[k]
    return d


# == Check Types # ==.....................................

# * I want to summarize all numerical like data types (float32, int64, etc.). But isinstance(np.float32, np.number) is false. So I need to check for each type individually
# * https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
NUMERICAL_TYPES = [
    int,
    float,
    complex,
    Decimal,
    np.number,
]
NUMERICAL_TYPES_NUMPY = [
    np.byte,
    np.ubyte,
    np.short,
    np.ushort,
    np.intc,
    np.uintc,
    np.int_,
    np.uint,
    np.longlong,
    np.ulonglong,
    np.half,
    np.single,
    np.double,
    np.longdouble,
    np.csingle,
    np.cdouble,
    np.clongdouble,
]


def get_type(o):
    """gets the type of an object as a string, e.g. 'int' or 'str'"""
    return str(type(o)).split("'")[1]


def check_type(o, allowed_types: list | tuple):
    """Checks if object if one of many types. Returns the type that matched.

    Args:
        o1 (_type_): _description_
        allowed_types (list | tuple): _description_

    Returns:
        _type_: _description_
    """
    result = [isinstance(o, t) for t in allowed_types]
    if not any(result):
        return False
    else:
        return allowed_types[result.index(True)]


### Ensure Type #.......................................................................................................
def ensure_tuple(
    s: str | list | tuple | str | Hashable | None, allow_none=True, convert_none=False
) -> tuple | None:
    """Converts Element into a tuple, even if it's just one"""
    if s is None:
        if convert_none:
            return (None,)
        elif allow_none:
            return None
        else:
            raise TypeError(f"#! Must pass tuple, None not allowed. ({s} was passed)")
    elif isinstance(s, tuple):
        return s
    elif isinstance(s, (list, set)):
        return tuple(s)
    else:
        return (s,)


def ensure_list(
    s: str | list | tuple | str | Hashable | None, allow_none=True, convert_none=False
) -> list | None:
    """Converts Element into a list, even if it's just one"""
    if s is None:
        if convert_none:
            return [
                None,
            ]
        elif allow_none:
            return None
        else:
            raise TypeError(f"#! Must pass tuple, None not allowed. ({s} was passed)")
    elif isinstance(s, list):
        return s
    elif isinstance(s, (tuple, set)):
        return list(s)
    else:
        return [
            s,
        ]


def ensure_nolen1(
    s: str | list | tuple | str | None, allow_none=True
) -> str | tuple | list:
    if s is None:
        if allow_none:
            return None
        else:
            raise TypeError(f"#! Must pass tuple, None not allowed. ({s}was passed)")
    elif len(s) == 1 and isinstance(s, (tuple, list, set)):
        return s[0]
    else:
        return s


def flatten(s: list | tuple, np=False) -> list:
    """

    :param s: Listor tuple or iterable
    :param np: If list is deeper than one Nesting (e.g. 3D), use numpy flatten method.
    Otherwise it'll perform list comprehension
    :return:
    """
    if np:
        return np.array(s, dtype="object").flatten(order="K").tolist()
    else:
        return [e for sublist in s for e in sublist]


def get_duplicate(s: list | tuple):
    result = [item for item, count in collections.Counter(s).items() if count > 1]
    if len(result):
        return result[0]
    else:
        return result


### Print & String #.......................................................................................................
def printable_dict(D, key_adjust=5, start_message=None, print_type=True):
    s = "\n######" + start_message if start_message else "\n######"
    for k, v in D.items():
        key = f"'{k}'" if type(k) is str else k
        val = f"'{v}'" if type(v) is str else v
        s += f"\n  |# {key.ljust(key_adjust)}: {val}"
        if print_type:
            s += f"\t {type(v)}"
        s += f"\t#|"
    s += "\n"
    return s


def capitalize(s: str) -> str:
    """Takes first word of string and capitalizes it, e.g. 'conc.: 1 mL'-> 'Conc.: 1 mL'"""
    # s = "conc.: 1 mL"
    s1 = s.split(" ")[0].capitalize()
    return s1 + " " + " ".join(s.split(" ")[1:])


# == CHECK IDENTITIY .......................................................................................................


def all_of_l1_in_l2(l1: list | tuple, l2: tuple | list) -> bool:
    """Checks if all elements of a list (x) are within another list (y)"""
    l1 = (l1,) if isinstance(l1, str) else l1
    l2 = (l2,) if isinstance(l2, str) else l2
    return all(tuple(e in l2 for e in l1))


def catch_duplicates(l: list | tuple) -> list:
    """Checks if all elements of a list (x) are within another list (y)"""
    l = (l,) if isinstance(l, str) else l
    return [item for item, count in collections.Counter(l).items() if count > 1]


def check_unordered_identity(
    o1: str | tuple | list, o2: str | tuple | list, ignore_duplicates=False
) -> bool:
    """both objects should be the same type. But we don't care about order of elements"""
    if not isinstance(o1, str) and not isinstance(o2, str):
        if set(o1) == set(o2):
            if len(o1) != len(o2) and not ignore_duplicates:
                dupl_o1 = catch_duplicates(o1)
                dupl_o2 = catch_duplicates(o2)
                raise AssertionError(
                    f"Both lists have matching elements, but at least one has duplicate elements: {dupl_o1}, {dupl_o2} "
                )
            return True
    elif isinstance(o1, int) and isinstance(o2, int):
        return o1 == o2
    elif isinstance(o1, str) and isinstance(o2, str):
        return o1 == o2
    else:
        raise AssertionError(
            f"#! {o1} is not comparable to {o2} (Did you miss the comma in your tuple?)"
        )


### REGULAR EXPRESSIONS #.......................................................................................................
def re_matchgroups(pattern, string: str, flags=None) -> list[dict]:
    """
    Takes a regular expression searchpattern that includes group names (?P<name>...)
    and returns a list of dictionaries with groupnames as keys and matched strings as values
    :param pattern: compiled re searchpattern, e.g. from re.compile(".*")
    :param string: str,
    :param flags: e.g. re.MULTILINE or re.DOTALL
    :returns dict

    :Example:

    >>> string = "abc abc2 abc3"
    >>> pattern = re.compile(r'(?P<WORD>abc)(?P<INDEX>\d)')
    >>> matches = re_matchgroups(pattern=pattern, string=string)
    >>> matches
    [{'WORD': 'abc', 'INDEX': '2'},
    {'WORD': 'abc', 'INDEX': '3'}]
    """
    return [
        match.groupdict()
        for match in re.finditer(pattern=pattern, string=string, flags=flags)
    ]


### OOP #.......................................................................................................


class OOP:
    """Helper functions for Object oriented programming"""

    def __str__(self):
        D = {
            a: getattr(self, a)
            for a in dir(self)
            if (not a.startswith("_") and not callable(getattr(self, a)))
        }

        ### Catch unprintable types
        if isinstance(D.get("data"), pd.DataFrame):
            D["data"] = (D["data"].shape, list(D["data"].columns))

        ### Make a new string
        return printable_dict(D=D, start_message=f"{type(self)}: ")

    # def __repr__(self):
    #     return str(self.__dict__)
    # def getobjdict(self):
    #     """ turns all the attributes into a dictionary, filtering out annoying dtypes"""
    #     # d = {"data": f"DataFrame shape {self.data.shape}"}
    #     # for a in dir(self):
    #     #     if (not a.startswith('__') and          # No dunders
    #     #             not callable(getattr(self, a)) and  # No methods
    #     #             not type(getattr(self, a)) in [type(pd), type(pd.DataFrame())] # Filter types
    #     #     ):
    #     #         d[a] = getattr(self, a)
    #     d = vars(self).copy() # Copy needed, since vars() returns a dictionary that can update an object's attribute!
    #     d["data"]= f"DataFrame shape {self.data.shape}"
    #     return d

    # def __copy__(self):
    #     """A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original."""
    #     cls = self.__class__ # __class__ returns Type of instance: <class '__main__.MKexp'>
    #     result = cls.__new__(cls) # __new__() creates intances and then calls __init__()
    #     result.__dict__.update(self.__dict__) # Update the attributes
    #     return result
    # def __deepcopy__(self, memo):
    #     """A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original."""
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for k, v in self.__dict__.items():
    #         setattr(result, k, deepcopy(v, memo))
    #     return result

    pass


# </editor-fold> ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^
