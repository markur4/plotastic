#
#%% imports

import os
import unittest
from glob import glob


# %% Gather tests

tests = glob("*_test.py")

# %% Run tests

import annotator_test

# %%
for test in tests:
    os.system(f"python {test}")