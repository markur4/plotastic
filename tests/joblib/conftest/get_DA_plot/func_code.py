# first line: 144
def get_DA_plot(dataset: str = "qpcr") -> plst.DataAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ### Load example data
        DF, dims = plst.load_dataset(dataset, verbose=False)

        ### Init DA
        DA = plst.DataAnalysis(DF, dims, subject="subject", verbose=False)

        DA.plot_box_strip()
        plt.close()
        return DA
