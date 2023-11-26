# first line: 163
def make_DA_all(dataset: str) -> plst.DataAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        DA = make_DA_statistics(dataset)
        DA.plot_box_swarm()
        plt.close()
        return DA
