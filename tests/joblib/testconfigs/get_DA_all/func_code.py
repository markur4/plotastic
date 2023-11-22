# first line: 161
def get_DA_all(dataset: str) -> plst.DataAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        DA = get_DA_statistics(dataset)
        DA.plot_box_swarm()
        plt.close()
        return DA
