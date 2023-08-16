import pandas as pd

from assumptions import Assumptions
from omnibus import Omnibus
from posthoc import PostHoc


# class StatResult(Assumptions, Omnibus, PostHoc):
#     def __init__(self, data: pd.DataFrame, dims: dict, verbose=False):
#         super().__init__(data=data, dims=dims, verbose=verbose)
