import pandas as pd

from plotastic.plottool import PlotTool


class MultiPlot(PlotTool):
    def __init__(self, **analysis_kws):
        super().__init__(**analysis_kws)
