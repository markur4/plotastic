# @dataclass

from typing import Dict
from copy import copy, deepcopy


class Dims:
    def __init__(
        self,
        y: str = None,
        x: str = None,
        hue: str = None,
        row: str = None,
        col: str = None,
    ):
        self.y = y
        self.x = x
        self.hue = hue
        self.row = row
        self.col = col
        self._by = None
        # self._by_str = None

    @property
    def by(self) -> list[str] | None:
        if self._by:
            return self._by
        elif self.row and self.col:
            return [self.row, self.col]
        elif self.row:
            return [self.row]
        elif self.col:
            return [self.col]
        else:
            return None

    def asdict(self, incl_None=True, incl_by=True) -> dict:
        d = dict(y=self.y, x=self.x, hue=self.hue, row=self.row, col=self.col)
        if incl_by:
            d.update(dict(by=self.by))
        if not incl_None:
            d = {k: v for (k, v) in d.items() if (not v is None)}
        return d

    def set(self, inplace=False, **kwargs) -> "Dims":
        newobj = self if inplace else copy(self)
        for k, v in kwargs.items():
            v = v if not v == "none" else None
            setattr(newobj, k, v)
        return newobj

    def getvalues(self, keys: list[str] | tuple[str], *args):
        """
        Converts a list of dimensions into a list of dimension values, e.g.
        :param keys: ["x", "y", "col"]
        :return: e.g. ["smoker", "tips", "day"]
        """
        defkeys = ("x", "y", "hue", "row", "col")
        l = []
        keys = [keys] + [arg for arg in args]
        for key in keys:
            assert key in defkeys, f"#! '{key}' should have been one of {defkeys}"
            l.append(getattr(self, key))
        return l

    def switch(
        self, *keys: str, inplace=False, verbose=True, **kwarg: str | Dict[str, str]
    ) -> "Dims":
        """
        Set attributes. Detects Duplicates, switches automatically
        :param keys: Two dimensions to switch. Only 2 Positional arguments allowed. Use e.g. dims.switch("x", "hue", **kwargs)
        :param inplace: Decide if this switching should change the dims object permanently (analogously to pandas dataframe). If False, you should pass return value into a variable
        :param verbose: Whether to print out switched values
        :param kwarg: e.g. dict(row="smoker")
        :return: dims object with switched parameters
        """

        """HANDLE ARGUMENTS if keys are passed, e.g. dims.switch("x","row",**kwargs)"""
        if len(keys) == 0:
            pass
        elif len(keys) == 2:
            assert len(kwarg) == 0, "#! Can't switch when both keys and kwarg is passed"
            values = self.getvalues(*keys)
            kwarg[keys[0]] = values[1]
        else:
            raise AssertionError(f"#! '{keys}' should have been of length 2")
        assert len(kwarg) == 1, f"#! {kwarg} should be of length 1 "

        """PRINT FIRST LINE"""
        if verbose:
            todo = "RE-WRITING" if inplace else "TEMPORARY CHANGING:"
            print(
                f"#! {todo} {self.__class__.__name__} with keys: '{keys}' and kwarg: {kwarg}:"
            )
            print("   (dim =\t'old' -> 'new')")

        ### SWITCH IT
        ### COPY OBJECT
        oldby = self.by
        original: dict = deepcopy(
            self.asdict(incl_None=True),
        )
        newobj = self if inplace else deepcopy(self)

        qK, qV = *kwarg.keys(), *kwarg.values()
        replace_v = "none"
        for oK, oV in original.items():  # Original Object
            if qK == oK:
                replace_v = oV
                setattr(newobj, qK, qV)
            elif qK != oK and oV == qV:
                replace_v = original[qK]
                setattr(newobj, oK, replace_v)
        assert (
            replace_v != "none"
        ), f"#! Did not find {list(kwarg.keys())} in dims {list(original.keys())}"

        ### PRINT THE OVERVIEW OF THE NEW MAPPING"""
        if verbose:
            for (oK, oV), nV in zip(original.items(), newobj.asdict().values()):
                pre = "  "
                if oV != nV and oV == replace_v:  # or replace_v == "none":
                    printval = f"'{replace_v}' -> '{qV}'"
                    pre = ">>"
                elif oK == "by" and newobj.by != oldby:
                    printval = (
                        f"'{oldby}' -> '{newobj.by}'"
                        if type(newobj.by) is str
                        else f"{oldby} -> {newobj.by}"
                    )
                elif oV != nV and oV != replace_v:
                    printval = f"'{oV}' -> '{replace_v}'"
                    pre = " <"
                else:  # oV == nV
                    printval = f"'{oV}'" if type(oV) is str else f"{oV}"
                if len(oK) < 3:
                    oK = oK + "  "

                printval = printval.replace("'None'", "None")  # REMOVE QUOTES

                print(f" {pre} {oK} =\t{printval}")

        ### x AND y MUST NOT BE None"""
        assert not None in [self.y, self.x], "#! This switch causes x or y to be None"

        return newobj
