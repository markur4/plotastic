import pandas as pd

from plotastic.plottool import PlotTool


class MultiPlot(PlotTool):
    def __init__(self, **dataframetool_kws):
        super().__init__(**dataframetool_kws)
