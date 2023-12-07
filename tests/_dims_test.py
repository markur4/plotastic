#
# %% imports
import plotastic as plst
import unittest


# %% Import Test Data
DF, dims = plst.load_dataset("tips")  #' Import Data
DA = plst.DataAnalysis(
    data=DF, dims=dims, title="tips"
)  #' Make DataAnalysis Object


# %% Unit Tests


class TestDataAnalysis(unittest.TestCase):
    def test_switching(self):
        v = False
        data, dims = plst.load_dataset("tips", verbose=v)
        DA = plst.DataAnalysis(data, dims, verbose=v)

        ### Chaining work?
        x, E1 = DA.dims.x, "size-cut"
        x_inchain, E2 = DA.switch("x", "hue", verbose=v).dims.x, "smoker"
        x_after_chaining, E3 = DA.dims.x, "size-cut"
        print(x, x_inchain, x_after_chaining)
        print(x != x_inchain)
        print(x == x_after_chaining)

        self.assertEqual(x, E1)
        self.assertEqual(x_inchain, E2)
        self.assertEqual(x_after_chaining, E3)


# %% __name__ == "__main__"

if __name__ == "__main__":
    unittest.main()
